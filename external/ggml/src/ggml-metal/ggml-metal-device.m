#import "ggml-metal-device.h"
#import "ggml-metal-ops.h" // ggml_metal_op_flash_attn_ext_use_vec_w64

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <errno.h>
#include <string.h>  // strcmp
#include <strings.h> // strcasecmp
#include <unistd.h>  // sysconf

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

static bool ggml_metal_op_bin_default_supported(const struct ggml_tensor * op) {
    return ggml_is_contiguous_rows(op->src[0]) &&
        ggml_is_contiguous_rows(op->src[1]) &&
        op->src[0]->type == GGML_TYPE_F32;
}

static bool ggml_metal_op_bin_bcast_supported(const struct ggml_tensor * op) {
    const bool src0_broadcast = !ggml_are_same_shape(op->src[0], op) && ggml_can_repeat(op->src[0], op);
    const bool src1_broadcast = !ggml_are_same_shape(op->src[1], op) && ggml_can_repeat(op->src[1], op);
    const bool supported_type =
        (op->type == GGML_TYPE_F32 &&
            op->src[0]->type == GGML_TYPE_F32 &&
            op->src[1]->type == GGML_TYPE_F32) ||
        (op->type == GGML_TYPE_F16 &&
            op->src[0]->type == GGML_TYPE_F16 &&
            op->src[1]->type == GGML_TYPE_F16);

    return (op->op == GGML_OP_ADD || op->op == GGML_OP_SUB) &&
        supported_type &&
        (ggml_are_same_shape(op->src[0], op) || src0_broadcast) &&
        (ggml_are_same_shape(op->src[1], op) || src1_broadcast) &&
        !ggml_metal_op_bin_default_supported(op);
}

static bool ggml_metal_op_pad_left_supported(const struct ggml_tensor * op) {
    const bool is_supported_type =
        (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) ||
        (op->src[0]->type == GGML_TYPE_F16 && op->type == GGML_TYPE_F16) ||
        (op->src[0]->type == GGML_TYPE_I32 && op->type == GGML_TYPE_I32);
    const bool has_left_padding =
        ggml_get_op_params_i32(op, 0) > 0 ||
        ggml_get_op_params_i32(op, 2) > 0 ||
        ggml_get_op_params_i32(op, 4) > 0 ||
        ggml_get_op_params_i32(op, 6) > 0;

    return op->op == GGML_OP_PAD &&
        ggml_get_op_params_i32(op, 8) == 0 &&
        has_left_padding &&
        is_supported_type;
}

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

struct ggml_metal_library {
    id<MTLLibrary> obj;

    ggml_metal_device_t dev;
    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif

        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

                if (ggml_metal_device_get_props(dev)->has_tensor) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
                }

                // compile the kernels for the SIMD group width this device actually executes.
                // ggml-metal.metal defaults to 32 when this is not supplied, so Apple silicon is unchanged
                [prep setObject:[NSString stringWithFormat:@"%d", ggml_metal_device_get_props(dev)->simd_width]
                         forKey:@"N_SIMDWIDTH"];

#if GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;

                //[options setFastMathEnabled:false];

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj       = library;
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        [prep setObject:[NSString stringWithFormat:@"%d", ggml_metal_device_get_props(dev)->simd_width]
                 forKey:@"N_SIMDWIDTH"];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->obj       = library;
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
    return lib->dev;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [lib->obj newFunctionWithName:base_func];
        } else {
            mtl_function = [lib->obj newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        // the whole library is compiled with a single N_SIMDWIDTH, taken from a one-off probe of a
        // trivial kernel. Metal is free to pick the execution width per pipeline (on RDNA parts it
        // picks wave32 or wave64 depending on register pressure), so a real kernel can disagree
        // with the probe - and every lane -> data mapping in that kernel would then be wrong.
        // refuse loudly rather than compute silent garbage.
        //
        // note: this guard aborts where its sibling immediately above returns a null pipeline.
        //       they differ deliberately. the sibling is upstream's own sanity check on a
        //       degenerate PSO and is left exactly as upstream wrote it, so Apple-silicon
        //       behaviour is untouched. this one is the invariant the whole wave64 port rests
        //       on: there is no configuration to continue into, because every kernel in the
        //       library has already been compiled for a lane count the hardware is not using.
        //
        //       it therefore aborts *before* releasing obj or unlocking. cleaning up first would
        //       write a fatal path as if it were recoverable - and if GGML_ASSERT were ever
        //       compiled out, the code below would publish a released pipeline object and the
        //       function tail would unlock a lock it no longer holds.
        {
            const int tew = (int) obj.threadExecutionWidth;
            const int nsw = ggml_metal_device_get_props(lib->dev)->simd_width;

            if (tew != nsw) {
                GGML_LOG_ERROR("%s: pipeline '%s' (base = '%s') has threadExecutionWidth = %d, but the library was compiled with N_SIMDWIDTH = %d\n",
                        __func__, name, base, tew, nsw);
                GGML_LOG_ERROR("%s: the lane -> data mapping in this kernel is wrong at %d lanes - refusing the pipeline\n",
                        __func__, tew);

                GGML_ASSERT(tew == nsw && "Metal pipeline threadExecutionWidth does not match the compiled N_SIMDWIDTH");
            }
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
};

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    [encoder->obj endEncoding];
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // the SIMD group width as the hardware probe reported it, raw: 0 means the probe failed or
    // returned a width this backend does not support, in which case props.simd_width holds the
    // 32-lane fallback instead. capability decisions that must reflect the real hardware
    // (has_mm_w64) key off this field, so the fallback can never arm a wave64-only path.
    int simd_width_probed;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

ggml_metal_rsets_t ggml_metal_rsets_init(void) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, 2*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  // half a second
                  usleep(500 * 1000);
              }
        }
#endif
    });

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

// read an integer environment dial.
//
// returns false and leaves *out untouched when the variable is unset or not a valid integer, so a
// typo can never silently select a degenerate configuration. values outside [lo, hi] are clamped.
// whatever ends up being applied is logged.
static bool ggml_metal_getenv_int(const char * name, int lo, int hi, int * out) {
    const char * val = getenv(name);

    if (val == NULL) {
        return false;
    }

    char * end = NULL;

    errno = 0;
    const long v = strtol(val, &end, 10);

    if (end == val || *end != '\0' || errno == ERANGE) {
        GGML_LOG_WARN("%s: ignoring %s='%s' - not a valid integer, keeping the default\n", __func__, name, val);
        return false;
    }

    long res = v;

    if (res < lo || res > hi) {
        res = res < lo ? lo : hi;
        GGML_LOG_WARN("%s: %s=%ld is outside the supported range [%d, %d] - clamping to %ld\n", __func__, name, v, lo, hi, res);
    }

    GGML_LOG_INFO("%s: %s = %ld\n", __func__, name, res);

    *out = (int) res;

    return true;
}

ggml_metal_device_t ggml_metal_device_init(int device) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
        dev->mtl_device = MTLCreateSystemDefaultDevice();

        // on multi-GPU systems the system default device is not necessarily the most capable one - for
        // example, on Intel Macs it is the GPU driving the main display, which can be the weaker card.
        // GGML_METAL_DEVICE selects a device explicitly, either by index into MTLCopyAllDevices() or by
        // a case-insensitive substring of the device name
        {
            const char * sel = getenv("GGML_METAL_DEVICE");

            if (sel != NULL) {
                NSArray * devices = MTLCopyAllDevices();

                id<MTLDevice> selected = nil;

                char * end = NULL;
                const long idx = strtol(sel, &end, 10);

                if (end != sel && *end == '\0') {
                    if (idx >= 0 && idx < (long) [devices count]) {
                        selected = [devices objectAtIndex:idx];
                    }
                } else {
                    NSString * name = [NSString stringWithUTF8String:sel];

                    for (id<MTLDevice> cur in devices) {
                        if ([[cur name] rangeOfString:name options:NSCaseInsensitiveSearch].location != NSNotFound) {
                            selected = cur;
                            break;
                        }
                    }
                }

                if (selected != nil) {
                    [selected retain];
                    [dev->mtl_device release];
                    dev->mtl_device = selected;

                    GGML_LOG_INFO("%s: GGML_METAL_DEVICE='%s' selected device: %s\n", __func__, sel, [[selected name] UTF8String]);
                } else {
                    GGML_LOG_WARN("%s: GGML_METAL_DEVICE='%s' did not match any device - using the system default\n", __func__, sel);
                }

                [devices release];
            }
        }

        if (dev->mtl_device) {
            dev->mtl_queue = [dev->mtl_device newCommandQueue];
            if (dev->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            dev->addr_virt = 0x000000400ULL;

            dev->props.device = device;

            // determine the SIMD group width by compiling a trivial kernel and asking the PSO.
            // Apple GPUs report 32, AMD GCN parts report 64 - the shader library is then compiled
            // with a matching N_SIMDWIDTH so the lane -> data mapping in every kernel is correct.
            //
            // the width is deliberately not overridable: ggml_metal_library_compile_pipeline()
            // asserts every pipeline's threadExecutionWidth against it, so a width that agreed
            // with the probe would change nothing and one that disagreed would abort at the first
            // pipeline. the supported A/B dial for the GEMM is GGML_METAL_MM_W64_DISABLE=1, which
            // leaves the width at 64 and only changes which mat-mul path is selected.
            dev->props.simd_width  = 32;
            dev->simd_width_probed = 0;
            {
                NSError * perror = nil;

                id<MTLLibrary> plib = [dev->mtl_device newLibraryWithSource:@"kernel void ggml_probe(device float * dst [[buffer(0)]], uint tpig [[thread_position_in_grid]]) { dst[tpig] = 0.0f; }"
                                                                    options:nil
                                                                      error:&perror];
                if (!plib) {
                    GGML_LOG_ERROR("%s: SIMD width probe: failed to compile the probe library: %s\n", __func__,
                            perror ? [[perror description] UTF8String] : "unknown error");
                } else {
                    id<MTLFunction> pfun = [plib newFunctionWithName:@"ggml_probe"];
                    if (!pfun) {
                        GGML_LOG_ERROR("%s: SIMD width probe: failed to find the function 'ggml_probe' in the probe library\n", __func__);
                    } else {
                        id<MTLComputePipelineState> pps = [dev->mtl_device newComputePipelineStateWithFunction:pfun error:&perror];
                        if (!pps) {
                            GGML_LOG_ERROR("%s: SIMD width probe: failed to create the probe pipeline state: %s\n", __func__,
                                    perror ? [[perror description] UTF8String] : "unknown error");
                        } else {
                            dev->simd_width_probed = (int) pps.threadExecutionWidth;
                            [pps release];
                        }
                        [pfun release];
                    }
                    [plib release];
                }

                if (dev->simd_width_probed == 32 || dev->simd_width_probed == 64) {
                    dev->props.simd_width = dev->simd_width_probed;
                } else {
                    if (dev->simd_width_probed != 0) {
                        GGML_LOG_ERROR("%s: SIMD width probe: reported an unsupported width of %d - only 32 and 64 are supported\n",
                                __func__, dev->simd_width_probed);

                        dev->simd_width_probed = 0;
                    }

                    // falling through to 32 here is the *old* behaviour and it is wrong on wave64
                    // hardware - say so, because the failure mode is out-of-bounds kernels.
                    GGML_LOG_ERROR("%s: SIMD width probe: no usable width - falling back to %d, which is incorrect on wave64 GPUs\n",
                            __func__, dev->props.simd_width);
                }
            }

            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            // wave64 GPUs have no simdgroup_matrix support, so they get the register-tiled
            // kernel_mul_mm_w64 instead. GGML_METAL_MM_W64_DISABLE falls back to the mat-vec path.
            // note: the raw hardware probe is a necessary condition, so the 32-lane fallback that
            //       a failed probe leaves in props.simd_width can never arm the wave64 GEMM. the
            //       compiled width is required to be 64 as well - today the two can only differ
            //       when the probe failed, but stating both keeps a 64-lane kernel from ever
            //       being paired with a 32-lane library.
            dev->props.has_mm_w64 =
                !dev->props.has_simdgroup_mm &&
                dev->simd_width_probed == 64 &&
                dev->props.simd_width  == 64;
            // value-aware, unlike a bare getenv() presence test: this is the dial the kernel
            // comment advertises for A/B measurement, and MM_W64_DISABLE=0 disabling the GEMM
            // would silently invert an experiment. "", "0", "false", "no", "off" all mean off.
            {
                const char * s = getenv("GGML_METAL_MM_W64_DISABLE");
                if (s && !(s[0] == 0 ||
                           strcmp(s, "0") == 0 ||
                           strcasecmp(s, "false") == 0 ||
                           strcasecmp(s, "no") == 0 ||
                           strcasecmp(s, "off") == 0)) {
                    if (dev->props.has_mm_w64) {
                        GGML_LOG_INFO("%s: GGML_METAL_MM_W64_DISABLE=%s - wave64 mat-mul off, using mat-vec\n", __func__, s);
                    }
                    dev->props.has_mm_w64 = false;
                }
            }

            // wave64 devices also get their own flash-attention vec kernels
            // (kernel_flash_attn_ext_vec_w64 / _vec_reduce_w64 in ggml-metal.metal), which are
            // only compiled into the library when N_SIMDWIDTH == 64. same necessary conditions
            // as the GEMM: the raw probe must have said 64 so a failed probe's 32-lane fallback
            // can never arm a 64-lane kernel, and the compiled width must agree.
            //
            // this prop says "the kernels exist". whether a given node can use them is a
            // per-node shape question - see ggml_metal_op_flash_attn_ext_use_vec_w64().
            dev->props.has_fa_vec_w64 =
                !dev->props.has_simdgroup_mm &&
                dev->simd_width_probed == 64 &&
                dev->props.simd_width  == 64;
            // value-aware, like GGML_METAL_MM_W64_DISABLE and unlike a bare presence test:
            // FA_W64_DISABLE=0 silently inverting an A/B is exactly the failure this avoids.
            // "", "0", "false", "no", "off" all mean off.
            {
                const char * s = getenv("GGML_METAL_FA_W64_DISABLE");
                if (s && !(s[0] == 0 ||
                           strcmp(s, "0") == 0 ||
                           strcasecmp(s, "false") == 0 ||
                           strcasecmp(s, "no") == 0 ||
                           strcasecmp(s, "off") == 0)) {
                    if (dev->props.has_fa_vec_w64) {
                        GGML_LOG_INFO("%s: GGML_METAL_FA_W64_DISABLE=%s - wave64 flash-attention off, FLASH_ATTN_EXT falls back to another backend\n", __func__, s);
                    }
                    dev->props.has_fa_vec_w64 = false;
                }
            }

            // mat-vec -> mat-mul crossover. 8 and 32 are the upstream defaults, tuned on Apple
            // GPUs; on this card the fork measured batching only becoming cheap above ~32, so
            // these want sweeping per device rather than inheriting.
            //
            // note: 8 is unmeasured on this card and wants a sweep - but the dials may only move
            //       the crossover, never remove it: mm_min = 0 would push every single-token
            //       mat-vec through the GEMM.
            dev->props.mm_min    = 8;
            dev->props.mm_id_min = 32;

            ggml_metal_getenv_int("GGML_METAL_MM_MIN",    1, 4096, &dev->props.mm_min);
            ggml_metal_getenv_int("GGML_METAL_MM_ID_MIN", 1, 4096, &dev->props.mm_id_min);

            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
            if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                dev->props.has_bfloat = false;
            }

            dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                dev->props.has_tensor = false;
            }

            // note: disable the tensor API by default for old chips because with the current implementation it is not useful
            // - M2 Ultra:   ~5% slower
            // - M4, M4 Max: no significant difference
            //
            // TODO: try to update the tensor API kernels to at least match the simdgroup performance
            if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                ![[dev->mtl_device name] containsString:@"M5"] &&
                ![[dev->mtl_device name] containsString:@"M6"] &&
                ![[dev->mtl_device name] containsString:@"A19"] &&
                ![[dev->mtl_device name] containsString:@"A20"]) {
                GGML_LOG_INFO("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                dev->props.has_tensor = false;
            }

            // double-check that the tensor API compiles
            if (dev->props.has_tensor) {
                const char * src_tensor_f16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                    dev->props.has_tensor = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
            if (dev->props.has_tensor && dev->props.has_bfloat) {
                const char * src_tensor_bf16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                    dev->props.has_bfloat = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            // In case of eGPU, shared memory may be preferable.
            dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.op_offload_min_batch_size  = 32;

            ggml_metal_getenv_int("GGML_OP_OFFLOAD_MIN_BATCH", 1, 1048576, &dev->props.op_offload_min_batch_size);

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
            if (@available(macOS 10.12, iOS 16.0, *)) {
                dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
            } else {
                dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
            }

            snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
            snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", [[dev->mtl_device name] UTF8String]);

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            if (dev->props.use_residency_sets) {
                dev->rsets = ggml_metal_rsets_init();
            } else {
                dev->rsets = nil;
            }

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: simd group width      = %d\n", __func__, dev->props.simd_width);
            GGML_LOG_INFO("%s: wave64 mat-mul        = %s\n", __func__, dev->props.has_mm_w64             ? "true" : "false");
            GGML_LOG_INFO("%s: wave64 flash attn     = %s\n", __func__, dev->props.has_fa_vec_w64         ? "true" : "false");
            GGML_LOG_INFO("%s: mat-mul min batch     = %d (id: %d)\n", __func__, dev->props.mm_min, dev->props.mm_id_min);
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    ggml_metal_rsets_free(dev->rsets);

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, 2*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    [event release];

    free(ev);

    GGML_UNUSED(dev);
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
    if (!res) {
        GGML_ABORT("%s: failed to wait for event\n", __func__);
    }

    GGML_UNUSED(dev);
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

// the iq1_* / iq2_* / iq3_* mat-vec kernels hardcode a 32-lane SIMD group: they seed threadgroup
// scratch with `pos = (32*sgitg + tiisg)*nval` and stride their block loop by a literal 32. at 64
// lanes tiisg reaches 63, so pos runs past both the threadgroup allocation and the constant grid
// tables - an out-of-bounds write and an out-of-bounds read - and the block loop double-counts.
// refuse these types here so the tensor falls back to another backend (CPU) instead of faulting.
//
// iq4_nl and iq4_xs are wave-generic (they index with N_SIMDWIDTH) and stay supported.
static bool ggml_metal_type_supported_at_simd_width(ggml_metal_device_t dev, enum ggml_type type) {
    if (dev->props.simd_width == 32) {
        return true;
    }

    switch (type) {
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
            break;
        default:
            return true;
    }

    static atomic_flag warned = ATOMIC_FLAG_INIT;

    if (!atomic_flag_test_and_set(&warned)) {
        GGML_LOG_WARN("%s: refusing %s tensors: the iq1_*/iq2_*/iq3_* mat-vec kernels assume a 32-wide SIMD group and this device is %d-wide. these ops fall back to another backend\n",
                __func__, ggml_type_name(type), dev->props.simd_width);
    }

    return false;
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_fa_vec_w64          = dev->props.has_fa_vec_w64;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;
    const int  simd_width              = dev->props.simd_width;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONCAT:
            // kernel_concat is instantiated for these element types only (kernel_concat_<type>)
            return op->src[0]->type == op->type && op->src[1]->type == op->type &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 ||
                    op->type == GGML_TYPE_I32 || op->type == GGML_TYPE_I16);
        case GGML_OP_ADD:
        case GGML_OP_SUB:
            return ggml_metal_op_bin_default_supported(op) ||
                ggml_metal_op_bin_bcast_supported(op);
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_COL2IM_1D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->type == op->src[0]->type &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    op->src[0]->type == GGML_TYPE_F16 ||
                    op->src[0]->type == GGML_TYPE_BF16);
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_FAST_1D:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (ggml_is_contiguous(op->src[1]) || ggml_is_contiguous_channels(op->src[1]));
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }

            return ((ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0)) ||
                ggml_metal_op_pad_left_supported(op);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
        case GGML_OP_ROLL:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    break;
                case GGML_TYPE_BF16:
                    if (!has_bfloat) {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
            if (has_simdgroup_mm) {
                return true;
            }

            // wave64 devices have no simdgroup_matrix, so kernel_flash_attn_ext_impl and
            // kernel_flash_attn_ext_blk stay out of reach and prefill-shaped nodes keep
            // answering no. the decode-shaped vec nodes do have a wave64 kernel
            // (kernel_flash_attn_ext_vec_w64), so admit exactly the shapes it is instantiated
            // for - by the same predicate ggml_metal_op_flash_attn_ext() uses to select it,
            // so the two can never disagree.
            //
            // Apple silicon never reaches this line: has_simdgroup_mm is true there and
            // has_fa_vec_w64 requires it to be false.
            return has_fa_vec_w64 && ggml_metal_op_flash_attn_ext_use_vec_w64(op);
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            {
                // the kernel processes the head dimension one SIMD group at a time and asserts
                // ne20 % GGML_METAL_NW(lib) == 0, so the capability check has to use the same width
                if (!has_simdgroup_reduction || op->src[2]->ne[0] % simd_width != 0) {
                    return false;
                }

                // ... and only kernel_gated_delta_net_f32_{1,2,4} are instantiated in
                // ggml-metal.metal, so the simdgroup count has to land in that set. a shape with,
                // say, ne20 = 192 at width 64 (or ne20 = 96 at width 32) gives nsg = 3, whose
                // kernel name does not exist: the pipeline comes back NULL and the encoder
                // dereferences it. refuse it here so the op falls back to another backend.
                const int64_t nsg = op->src[2]->ne[0]/simd_width;

                return nsg == 1 || nsg == 2 || nsg == 4;
            }
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            if (!ggml_metal_type_supported_at_simd_width(dev, op->src[0]->type)) {
                return false;
            }
            return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q1_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type != GGML_TYPE_NVFP4;
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op) &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    ggml_metal_device_rsets_rm(buf->dev, buf->rset);

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(buf);

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

// wrap a host pointer in a zero-copy shared MTLBuffer
// newBufferWithBytesNoCopy requires a page-aligned address and length, so wrap the containing
// pages and report the offset of the data within the wrapped range. returns nil if the memory
// cannot be wrapped (e.g. malloc'd memory spanning multiple VM regions) - callers must fall
// back to a staged copy in that case
static id<MTLBuffer> ggml_metal_buffer_wrap_host(id<MTLDevice> device, const void * data, size_t size, size_t * offs) {
    // escape hatch for bisecting: force the staged-copy path
    if (getenv("GGML_METAL_NO_ZEROCOPY") != NULL) {
        return nil;
    }

    const uintptr_t page = (uintptr_t) sysconf(_SC_PAGESIZE);

    const uintptr_t addr    = (uintptr_t) data;
    const uintptr_t addr_lo = addr & ~(page - 1);

    const size_t len = (size_t)(((addr - addr_lo) + size + page - 1) & ~(page - 1));

    *offs = (size_t)(addr - addr_lo);

    return [device newBufferWithBytesNoCopy:(void *)(uintptr_t) addr_lo
                                     length:len
                                    options:MTLResourceStorageModeShared
                                deallocator:nil];
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, bid_dst.offs + size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        size_t offs_src = 0;

        id<MTLBuffer> buf_src = ggml_metal_buffer_wrap_host(buf->dev->mtl_device, data, size, &offs_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // the host memory could not be wrapped zero-copy - stage the copy through a shared buffer
        if (buf_src == nil) {
            const size_t size_step = MIN(size, (size_t) 64u*1024*1024);

            id<MTLBuffer> buf_stg = [buf->dev->mtl_device newBufferWithLength:size_step options:MTLResourceStorageModeShared];
            GGML_ASSERT(buf_stg);

            for (size_t i = 0; i < size; i += size_step) {
                const size_t size_cur = MIN(size_step, size - i);

                memcpy([buf_stg contents], (const char *) data + i, size_cur);

                id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

                {
                    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                    [encoder copyFromBuffer:buf_stg
                               sourceOffset:0
                                   toBuffer:bid_dst.metal
                          destinationOffset:bid_dst.offs + i
                                       size:size_cur];

                    [encoder endEncoding];
                }

                [cmd_buf commit];
                [cmd_buf waitUntilCompleted];
            }

            [buf_stg release];

            return;
        }

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:offs_src
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

bool ggml_metal_buffer_cpy_tensor(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t)src->buffer->context;

    const size_t size = ggml_nbytes(src);

    // if both buffers are shared, we can use memcpy directly
    if (buf_dst->is_shared && buf_src->is_shared) {
        memcpy(dst->data, src->data, size);
        return true;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf_src, src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf_dst, dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }

    return true;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
