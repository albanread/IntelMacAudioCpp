#include "ggml-metal-device.h"

// SIMD group width of the library's device (32 on Apple Silicon, 64 on AMD GCN)
#define GGML_METAL_NW(lib) (ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width)

#include "ggml-metal-impl.h"

#include "ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(int device) {
    static std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device));

    return devs.back().get();
}

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    if (!ppls) {
        return;
    }

    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        case GGML_OP_CONCAT: op_str = "concat"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_contig(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_contig_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_2d(ggml_metal_library_t lib, ggml_type type) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_2d_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_row_f32(ggml_metal_library_t lib) {
    const char * base = "kernel_cpy_row_f32";
    const char * name = "kernel_cpy_row_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_transpose_f32(ggml_metal_library_t lib) {
    const char * base = "kernel_cpy_transpose_f32";
    const char * name = "kernel_cpy_transpose_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, sizeof(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, ggml_type tidx, ggml_type tdst) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_set_rows_%s_%s", ggml_type_name(tdst), ggml_type_name(tidx));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int n = op->src[0]->ne[0];

    snprintf(base, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", base, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nsg  = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag_mask_inf(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_diag_mask_inf_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SCALE:      op_num = OP_UNARY_NUM_SCALE;      break;
        case GGML_OP_FILL:       op_num = OP_UNARY_NUM_FILL;       break;
        case GGML_OP_CLAMP:      op_num = OP_UNARY_NUM_CLAMP;      break;
        case GGML_OP_SQR:        op_num = OP_UNARY_NUM_SQR;        break;
        case GGML_OP_SQRT:       op_num = OP_UNARY_NUM_SQRT;       break;
        case GGML_OP_SIN:        op_num = OP_UNARY_NUM_SIN;        break;
        case GGML_OP_COS:        op_num = OP_UNARY_NUM_COS;        break;
        case GGML_OP_LOG:        op_num = OP_UNARY_NUM_LOG;        break;
        case GGML_OP_LEAKY_RELU: op_num = OP_UNARY_NUM_LEAKY_RELU; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_num = OP_UNARY_NUM_TANH;        break;
                case GGML_UNARY_OP_RELU:        op_num = OP_UNARY_NUM_RELU;        break;
                case GGML_UNARY_OP_SIGMOID:     op_num = OP_UNARY_NUM_SIGMOID;     break;
                case GGML_UNARY_OP_GELU:        op_num = OP_UNARY_NUM_GELU;        break;
                case GGML_UNARY_OP_GELU_ERF:    op_num = OP_UNARY_NUM_GELU_ERF;    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_num = OP_UNARY_NUM_GELU_QUICK;  break;
                case GGML_UNARY_OP_SILU:        op_num = OP_UNARY_NUM_SILU;        break;
                case GGML_UNARY_OP_ELU:         op_num = OP_UNARY_NUM_ELU;         break;
                case GGML_UNARY_OP_NEG:         op_num = OP_UNARY_NUM_NEG;         break;
                case GGML_UNARY_OP_ABS:         op_num = OP_UNARY_NUM_ABS;         break;
                case GGML_UNARY_OP_SGN:         op_num = OP_UNARY_NUM_SGN;         break;
                case GGML_UNARY_OP_STEP:        op_num = OP_UNARY_NUM_STEP;        break;
                case GGML_UNARY_OP_HARDSWISH:   op_num = OP_UNARY_NUM_HARDSWISH;   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_num = OP_UNARY_NUM_HARDSIGMOID; break;
                case GGML_UNARY_OP_EXP:         op_num = OP_UNARY_NUM_EXP;         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_num = OP_UNARY_NUM_SOFTPLUS;    break;
                case GGML_UNARY_OP_EXPM1:       op_num = OP_UNARY_NUM_EXPM1;       break;
                case GGML_UNARY_OP_FLOOR:       op_num = OP_UNARY_NUM_FLOOR;       break;
                case GGML_UNARY_OP_CEIL:        op_num = OP_UNARY_NUM_CEIL;        break;
                case GGML_UNARY_OP_ROUND:       op_num = OP_UNARY_NUM_ROUND;       break;
                case GGML_UNARY_OP_TRUNC:       op_num = OP_UNARY_NUM_TRUNC;       break;
                case GGML_UNARY_OP_XIELU:       op_num = OP_UNARY_NUM_XIELU;       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;
    const bool is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SUM_ROWS: op_num = OP_SUM_ROWS_NUM_SUM_ROWS; break;
        case GGML_OP_MEAN:     op_num = OP_SUM_ROWS_NUM_MEAN;     break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(base, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", base, op_num);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_SUM_ROWS + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = GGML_METAL_NW(lib)*sizeof(float);

    if (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "tri";
    const int ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = GGML_METAL_NW(lib)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(ggml_metal_library_t lib, const ggml_tensor * op, int ssm_conv_bs) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d", base, ssm_conv_bs);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, const ggml_tensor * op)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    char base[256];
    char name[256];

    const int nsg = (ne00 + GGML_METAL_NW(lib) - 1)/GGML_METAL_NW(lib);

    snprintf(base, 256, "kernel_ssm_scan_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (32 + 2) floats
    res.smem = (GGML_METAL_NW(lib) + 2)*sizeof(float)*nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    const int ne20 = op->src[2]->ne[0]; // S_v
    const int ne21 = op->src[2]->ne[1]; // H
    const int ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 3D (S_v*S_v*H, K, n_seqs); K is the snapshot slot count.
    const int K = op->src[5]->ne[1];

    const int nsg = op->src[2]->ne[0]/GGML_METAL_NW(lib);

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);

    // the kernel walks S_v one SIMD group at a time, so ne20 must be a whole number of them.
    // ggml_metal_device_supports_op() applies the same `ne20 % simd_width` test against the same
    // probed width, so the scheduler should never route a shape that fails here - in particular
    // ne20 = 32 at width 64, which used to reach this function and produce nsg = 0.
    // ne20 = 0 still satisfies the modulo and would give nsg = 0 and a kernel name that does not
    // exist, so require the simdgroup count explicitly rather than inferring it.
    GGML_ASSERT(ne20 % GGML_METAL_NW(lib) == 0);

    // ggml-metal.metal instantiates kernel_gated_delta_net_f32_{1,2,4} and nothing else, so nsg
    // has to land in that set - `nsg >= 1` is not enough. nsg = 3 (ne20 = 192 at width 64, or
    // ne20 = 96 at width 32) names a kernel that does not exist: newFunctionWithName returns nil,
    // compile_pipeline logs and hands back a NULL pipeline, and ggml_metal_op_gated_delta_net
    // then dereferences it inside ggml_metal_encoder_set_pipeline.
    // ggml_metal_device_supports_op() refuses those shapes so they route to another backend and
    // never reach here; this assert is the belt-and-braces copy of that test.
    GGML_ASSERT((nsg == 1 || nsg == 2 || nsg == 4) &&
            "only kernel_gated_delta_net_f32_{1,2,4} are instantiated in ggml-metal.metal");

    snprintf(base, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d", base, ne20, ne30, K);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_set_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_set_int16(cv, K,    FC_GATED_DELTA_NET + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int nsg = 8;
    const int n   = op->src[1]->ne[1];
    const int k   = op->src[1]->ne[0];

    snprintf(base, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d", base, nsg, n, k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_set_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_set_int16(cv, k,   FC_SOLVE_TRI + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, GGML_METAL_NW(lib))*nsg*sizeof(float), 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nxpsg, int r1ptg) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;
    const int       ne12  = op->src[1]->ne[2];
    const int       r2    = ne12 / op->src[0]->ne[2];
    const int       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

// ---------------------------------------------------------------------------------------
// host-side mirror of the kernel_mul_mm_w64 tile dials.
//
// the kernel cannot see these names and the host cannot see the kernel's, so the contract is
// held by three things that must be edited together: the single instantiation list in
// ggml-metal.metal, the static_asserts inside kernel_mul_mm_w64, and the asserts here.
// everything the host computes for the w64 path - the C tile, the threadgroup memory size, the
// bc_inp predicate, the thread count - is derived from these names rather than written as a
// literal, so moving a dial on one side and not the other is a build or startup failure rather
// than a silent out-of-bounds threadgroup write.
//
// these must match kernel_mul_mm_w64<..., MM_W64_NK, MM_W64_TN, MM_W64_UR> in ggml-metal.metal:
//
//   #define MM_W64_NR0 64               and the instantiated NK = 32, TN = 2
//   #define MM_W64_TM   4
//   #define MM_W64_NR1 (16*MM_W64_TN)
// ---------------------------------------------------------------------------------------

static constexpr int MM_W64_NK  = 32;             // k staged per step
static constexpr int MM_W64_TM  = 4;              // rows of C per thread
static constexpr int MM_W64_TN  = 2;              // cols of C per thread
static constexpr int MM_W64_NR0 = 64;             // rows of C per threadgroup (M)
static constexpr int MM_W64_NR1 = 16*MM_W64_TN;   // cols of C per threadgroup (N)

// the kernel's thread -> tile mapping is hardwired to this many threads (4 wave64 simdgroups)
static constexpr int MM_W64_NUM_THREADS = 256;

static_assert((MM_W64_NR0/MM_W64_TM)*(MM_W64_NR1/MM_W64_TN) == MM_W64_NUM_THREADS,
        "the w64 C tile must be covered by exactly MM_W64_NUM_THREADS threads, one TM x TN patch each");
static_assert(MM_W64_NK % 16 == 0,
        "stage A dequantizes 16 k at a time, so a staging step must be a whole number of those - "
        "this is also what lets bc_inp be expressed as K % MM_W64_NK");

// size in bytes of ONE threadgroup staging element, for ONE operand. kernel_mul_mm_w64 has two
// independent staging types - S0 stages A (src0) into sa, S1 stages B (src1) into sb - and they
// are chosen per operand, not as a pair. keep this in step with the instantiation list in
// ggml-metal.metal.
//
// the rule: a staging type is float iff that operand's ggml type is F32, and half otherwise.
//
//   - f32 operand -> float. on wave64 the GEMM takes batched work that previously went to
//     kernel_mul_mv_t_t<float,float>, which was fp32 end to end (T1 yl[NF] with T1 = float).
//     staging an f32 operand through half would be a new, undeclared fp16 precision floor:
//     values outside +/-65504 flush to inf and anything needing sub-2^-14 resolution is lost.
//     it degrades quietly rather than failing, so it is not a trade to make silently - and it
//     applies to the weights of f32 x f16 just as much as to the activations of q8_0 x f32.
//   - f16 operand -> half. the value is already fp16 in memory, so widening the staging buffer
//     doubles its cost and recovers nothing.
//   - q8_0 src0 -> half. dequantize_q8_0 forms int8 * fp16-scale in fp32 and then casts to
//     S0_4x4, so with S0 = half the narrowing already happens inside the dequantize call;
//     that rounding costs ~2^-11 relative, well under q8_0's own ~2^-8 quantisation error.
//     keeping half also halves sa.
//
// a mismatch between this and the template arguments in ggml-metal.metal is an out-of-bounds
// threadgroup write, not a slow path: the kernel splits shmem at MM_W64_NR0*MM_W64_NK*sizeof(S0)
// while the host sizes the whole allocation.
static size_t ggml_metal_mul_mm_w64_stage_size(ggml_type t) {
    return t == GGML_TYPE_F32 ? sizeof(float) : sizeof(ggml_fp16_t);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;

    const struct ggml_metal_device_props * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    const bool has_tensor = props->has_tensor;

    // wave64 devices have no simdgroup_matrix, so they get the register-tiled GEMM instead
    const bool use_w64 = !has_tensor && props->has_mm_w64;

    // bounds-checked input path. for the simdgroup_matrix kernels the 32 is their own block
    // size; for the w64 kernel it is the staging step, because stage A reads a whole
    // 16-element vector per dequantize call with no per-element guard before the store, and
    // k_pos runs to the end of the last MM_W64_NK step rather than to K. both are 32 today.
    const bool bc_inp = use_w64
        ? (op->src[0]->ne[0] % MM_W64_NK != 0)
        : (op->src[0]->ne[0] % 32 != 0);

    // note: the w64 kernel bounds-checks every tile edge itself, so it has no bc_out variant
    const bool bc_out = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (!use_w64 && (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0));

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    const int16_t ne12 = (int16_t) op->src[1]->ne[2];
    const int16_t ne13 = (int16_t) op->src[1]->ne[3];
    const int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    const int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    snprintf(base, 256, use_w64 ? "kernel_mul_mm_w64_%s_%s" : "kernel_mul_mm_%s_%s",
             ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d",
             base, bc_inp, bc_out, ne12, ne13, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_set_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_set_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_set_int16(cv, r3,    FC_MUL_MM + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        // note: there is deliberately no fallback to kernel_mul_mm here. on wave64 hardware the
        //       simdgroup_matrix kernel does not merely underperform, it fails at PIPELINE
        //       CREATION ("call to an undefined label"), so silently degrading is not an option.
        //       ggml_metal_mul_mm_w64_supported() in ggml-metal-ops.cpp keeps types without a
        //       w64 variant on the mat-vec path, so reaching this assert means those two lists
        //       have drifted apart.
        GGML_ASSERT((res.pipeline || !use_w64) && "missing wave64 mat-mul variant for this type pair");

        ggml_metal_cv_free(cv);
    }

    if (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        const size_t smem_a = NRA * N_MM_NK_TOTAL * sizeof(ggml_fp16_t);
        res.smem = smem_a;
    } else if (use_w64) {
        // MM_W64_NR0 rows x MM_W64_NR1 cols of C per threadgroup (a 64 x 32 tile as shipped)
        res.nr0 = MM_W64_NR0;
        res.nr1 = MM_W64_NR1;

        // the same expression the kernel uses to lay out threadgroup memory:
        //   sa = shmem,                                   [MM_W64_NK][MM_W64_NR0] of S0
        //   sb = shmem + MM_W64_NR0*MM_W64_NK*sizeof(S0), [MM_W64_NK][MM_W64_NR1] of S1
        // the two halves are sized independently, one per operand, so sa and sb can differ:
        //
        //   variant     S0     S1       sa     sb    smem
        //   f32_f32     float  float  8192   4096   12288
        //   f16_f32     half   float  4096   4096    8192
        //   q8_0_f32    half   float  4096   4096    8192
        //   f32_f16     float  half   8192   2048   10240
        //   f16_f16     half   half   4096   2048    6144
        //
        // all five are under Metal's guaranteed 16 KB minimum; ggml_metal_op_mul_mat asserts
        // the result against this device's real max_theadgroup_memory_size before encoding.
        const size_t sz_a = ggml_metal_mul_mm_w64_stage_size(tsrc0);
        const size_t sz_b = ggml_metal_mul_mm_w64_stage_size(tsrc1);

        res.smem = (size_t) MM_W64_NK*MM_W64_NR0*sz_a
                 + (size_t) MM_W64_NK*MM_W64_NR1*sz_b;
    } else {
        res.nr0 = 64;
        res.nr1 = 32;

        res.smem = bc_out ? 8192 : (4096 + 2048);
    }

    res.nsg = N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    if (use_w64) {
        // ggml_metal_op_mul_mat dispatches GGML_METAL_NW(lib) x res.nsg threads, and every
        // thread -> tile mapping in kernel_mul_mm_w64 is derived from its NUM_THREADS == 256.
        // res.nsg comes from N_MM_SIMD_GROUP_X/Y, which are simdgroup_matrix tuning constants
        // with no connection to this kernel, so check the product rather than assume it.
        GGML_ASSERT(res.nsg*GGML_METAL_NW(lib) == MM_W64_NUM_THREADS &&
                "kernel_mul_mm_w64 requires exactly 256 threads per threadgroup");
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ne00 < GGML_METAL_NW(lib)) {
                    nsg = 1;
                    nr0 = GGML_METAL_NW(lib);
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = GGML_METAL_NW(lib)*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = GGML_METAL_NW(lib)*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    const int16_t r2 = (int16_t) (ne12 / ne02);
    const int16_t r3 = (int16_t) (ne13 / ne03);

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, r3,             FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1));
    snprintf(name, 256, "%s_bci=%d", base, bc_inp);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const char * suffix = "";

        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = GGML_METAL_NW(lib)*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = GGML_METAL_NW(lib)*sizeof(float)*N_R0_Q8_0;
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                nsg = N_SG_Q5_K;
                nr0 = N_R0_Q5_K;
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = GGML_METAL_NW(lib)*sizeof(float);
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s", ggml_type_name(tsrc0), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = GGML_METAL_NW(lib)*(sizeof(float) + sizeof(int32_t));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// note: reuse the argsort kernel for top_k
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    // do bounds checks for the mask?
    const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_mask,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg,
        bool    w64) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const int32_t ns10 = op->src[1]->nb[1]/op->src[1]->nb[0];
    const int32_t ns20 = op->src[2]->nb[1]/op->src[2]->nb[0];

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            w64 ? "flash_attn_ext_vec_w64" : "flash_attn_ext_vec",
            ggml_type_name(op->src[1]->type),
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT_VEC + 4);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg,
        bool    w64) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "%s", w64 ? "kernel_flash_attn_ext_vec_reduce_w64" : "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", base, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_num, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_bcast(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        default: GGML_ABORT("fatal error");
    };

    switch (op->type) {
        case GGML_TYPE_F32: snprintf(base, 256, "kernel_bin_bcast_f32"); break;
        case GGML_TYPE_F16: snprintf(base, 256, "kernel_bin_bcast_f16"); break;
        default: GGML_ABORT("fatal error");
    };
    snprintf(name, 256, "%s_op=%d", base, op_num);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    char base[256];
    char name[256];

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    snprintf(base, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.c4   = is_c4;
    res.smem = GGML_METAL_NW(lib)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = GGML_METAL_NW(lib)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = GGML_METAL_NW(lib)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE);

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d", base, is_imrope ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL || op->op == GGML_OP_IM2COL_FAST_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_COL2IM_1D);
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->type == op->src[0]->type);

    const char * suffix;
    switch (op->src[0]->type) {
        case GGML_TYPE_F32:  suffix = "f32";  break;
        case GGML_TYPE_F16:  suffix = "f16";  break;
        case GGML_TYPE_BF16: suffix = "bf16"; break;
        default: GGML_ABORT("col2im_1d: unsupported type");
    }

    char name[64];
    snprintf(name, sizeof(name), "kernel_col2im_1d_%s", suffix);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d_linear(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_linear_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(op->src[1]) || ggml_is_contiguous_channels(op->src[1]));

    char base[256];
    char name[256];

    const bool is_1d = op->ne[1] == 1 && op->src[1]->ne[1] == 1 && op->src[0]->ne[1] == 1 && op->ne[3] == 1;
    snprintf(base, 256, "kernel_conv_2d_dw_%s%s", is_1d ? "1d_" : "", ggml_is_contiguous(op->src[1]) ? "whcn" : "cwhn");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);

    if (mode == GGML_SCALE_MODE_BILINEAR) {
        snprintf(base, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        snprintf(base, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", base, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROLL);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }

    res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_left(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_left_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (res.pipeline) {
        return res;
    }

    res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) {
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    int nsg = 1;
    while (GGML_METAL_NW(lib)*nsg < ne00 && GGML_METAL_NW(lib)*nsg < 1024) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = GGML_METAL_NW(lib) * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}
