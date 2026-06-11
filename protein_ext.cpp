#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

extern "C" {
#include "esmf.h"
}

/* Hyperparameters & Structs  */

struct esmc_hparams {
    int   vocab_size     = 64;
    int   d_model        = 960;
    int   n_heads        = 15;
    int   head_dim       = 64;
    int   n_layers       = 30;
    int   ffn_hidden     = 2560;
    int   max_seq_len    = 2048;
    float scaling_factor = 0.9128709291752769f;
    float rope_base      = 10000.0f;
    int   rope_type      = 2;      /* GGML_ROPE_TYPE_NEOX */
};

struct esmc_layer {
    ggml_tensor *attn_ln_w;
    ggml_tensor *attn_ln_b;
    ggml_tensor *qkv_w;
    ggml_tensor *q_ln_w;
    ggml_tensor *k_ln_w;
    ggml_tensor *out_w;
    ggml_tensor *ffn_ln_w;
    ggml_tensor *ffn_ln_b;
    ggml_tensor *fc1_w;
    ggml_tensor *fc2_w;
};

struct esmc_model {
    esmc_hparams hparams;
    ggml_context *ctx_w;

    ggml_tensor *embed_w;
    std::vector<esmc_layer> layers;
    ggml_tensor *final_norm_w;

    ggml_tensor *lm_h0_w;
    ggml_tensor *lm_h0_b;
    ggml_tensor *lm_h2_w;
    ggml_tensor *lm_h2_b;
    ggml_tensor *lm_h3_w;
    ggml_tensor *lm_h3_b;
};

/* Tokenizer Vocabulary */

static const std::unordered_map<char,int> AA_TO_TOKEN = {
    {'L', 4},{'A', 5},{'G', 6},{'V', 7},
    {'S', 8},{'E', 9},{'R',10},{'T',11},
    {'I',12},{'D',13},{'P',14},{'K',15},
    {'Q',16},{'N',17},{'F',18},{'Y',19},
    {'M',20},{'H',21},{'W',22},{'C',23},
    {'X',24},{'B',25},{'U',26},{'Z',27},
    {'O',28},
};

static std::vector<int32_t> tokenise(const std::string &seq)
{
    std::vector<int32_t> tok;
    tok.push_back(0); // <cls>
    for (char c : seq) {
        char uc = (char)toupper((unsigned char)c);
        auto it = AA_TO_TOKEN.find(uc);
        if (it != AA_TO_TOKEN.end()) {
            tok.push_back(it->second);
        } else {
            tok.push_back(3); // <unk>
        }
    }
    tok.push_back(2); // <eos>
    return tok;
}

static const char *token_to_aa(int tok)
{
    static const char *vocab[] = {
        "<cls>","<pad>","<eos>","<unk>",
        "L","A","G","V","S","E","R","T","I","D","P","K",
        "Q","N","F","Y","M","H","W","C","X","B","U","Z","O",
    };
    if (tok >= 0 && tok < (int)(sizeof(vocab)/sizeof(vocab[0]))) {
        return vocab[tok];
    }
    return "?";
}

/* Weight Loader */

static int g_tensors_loaded = 0;

static ggml_type esmf_to_ggml(uint32_t dtype, const char *name)
{
    switch (dtype) {
        case ESMF_DTYPE_F32:  return GGML_TYPE_F32;
        case ESMF_DTYPE_F16:  return GGML_TYPE_F16;
        case ESMF_DTYPE_BF16: return GGML_TYPE_BF16;
        case ESMF_DTYPE_Q4_0: return GGML_TYPE_Q4_0;
        case ESMF_DTYPE_Q8_0: return GGML_TYPE_Q8_0;
        default:
            throw std::runtime_error(
                std::string("unsupported ESMF dtype for: ") + name);
    }
}

static ggml_tensor *load_tensor(ggml_context *ctx,
                                esmf_file_t *ef,
                                const char *name,
                                int ndim,
                                const int64_t *expected_shape)
{
    const esmf_tensor_t *et = esmf_find(ef, name);
    if (!et) {
        throw std::runtime_error(std::string("tensor not found: ") + name);
    }
    if (et->ndim != (uint32_t)ndim) {
        throw std::runtime_error(std::string("ndim mismatch for: ") + name);
    }
    if (ndim == 2) {
        if (et->shape[0] != expected_shape[1] || et->shape[1] != expected_shape[0]) {
            throw std::runtime_error(std::string("shape mismatch for 2D tensor: ") + name);
        }
    } else {
        for (int i = 0; i < ndim; i++) {
            if (et->shape[i] != expected_shape[i]) {
                throw std::runtime_error(std::string("shape mismatch for: ") + name);
            }
        }
    }

    ggml_type type = esmf_to_ggml(et->dtype, name);
    ggml_tensor *t = nullptr;
    if (ndim == 1) {
        t = ggml_new_tensor_1d(ctx, type, expected_shape[0]);
    } else if (ndim == 2) {
        t = ggml_new_tensor_2d(ctx, type, expected_shape[0], expected_shape[1]);
    } else {
        throw std::runtime_error("only 1D/2D tensors supported in weight loader");
    }

    assert(t);
    ggml_set_name(t, name);

    const void *src_data = esmf_data(ef, et);
    if (!src_data) {
        throw std::runtime_error(std::string("failed to map data for: ") + name);
    }
    memcpy(t->data, src_data, et->data_size);

    g_tensors_loaded++;
    return t;
}

static ggml_tensor *load_1d(ggml_context *ctx, esmf_file_t *ef, const char *name, int64_t d0)
{
    int64_t shape[1] = { d0 };
    return load_tensor(ctx, ef, name, 1, shape);
}

static ggml_tensor *load_2d(ggml_context *ctx, esmf_file_t *ef, const char *name, int64_t d0, int64_t d1)
{
    int64_t shape[2] = { d0, d1 };
    return load_tensor(ctx, ef, name, 2, shape);
}

static void esmc_load_weights(esmc_model &model, const char *path)
{
    esmf_file_t *ef = esmf_open(path);
    if (!ef) {
        throw std::runtime_error(std::string("failed to open ") + path);
    }

    const int D = model.hparams.d_model;
    const int V = model.hparams.vocab_size;
    const int F = model.hparams.ffn_hidden;

    /* Allocate weight context */
    struct ggml_init_params params = {
        1400 * 1024 * 1024, /* 1.4 GB weight budget */
        nullptr,
        false
    };
    model.ctx_w = ggml_init(params);
    if (!model.ctx_w) {
        esmf_close(ef);
        throw std::runtime_error("failed to initialize weight context");
    }

    model.embed_w = load_2d(model.ctx_w, ef, "token_embd.weight", D, V);

    model.layers.resize(model.hparams.n_layers);
    for (int i = 0; i < model.hparams.n_layers; i++) {
        esmc_layer &l = model.layers[i];
        char prefix[128];
        snprintf(prefix, sizeof(prefix), "blk.%d.", i);
        auto BLK = [&](const char *suffix) {
            static char name[256];
            snprintf(name, sizeof(name), "%s%s", prefix, suffix);
            return name;
        };

        l.attn_ln_w = load_1d(model.ctx_w, ef, BLK("attn_norm.weight"),   D);
        l.attn_ln_b = load_1d(model.ctx_w, ef, BLK("attn_norm.bias"),     D);
        l.qkv_w     = load_2d(model.ctx_w, ef, BLK("attn_qkv.weight"),    D, 3*D);
        l.q_ln_w    = load_1d(model.ctx_w, ef, BLK("attn_q_ln.weight"),   D);
        l.k_ln_w    = load_1d(model.ctx_w, ef, BLK("attn_k_ln.weight"),   D);
        l.out_w     = load_2d(model.ctx_w, ef, BLK("attn_out.weight"),    D, D);
        l.ffn_ln_w  = load_1d(model.ctx_w, ef, BLK("ffn_norm.weight"),    D);
        l.ffn_ln_b  = load_1d(model.ctx_w, ef, BLK("ffn_norm.bias"),      D);
        l.fc1_w     = load_2d(model.ctx_w, ef, BLK("ffn_gate_up.weight"), D, 2*F);
        l.fc2_w     = load_2d(model.ctx_w, ef, BLK("ffn_down.weight"),    F,   D);
    }

    model.final_norm_w = load_1d(model.ctx_w, ef, "output_norm.weight", D);

    model.lm_h0_w = load_2d(model.ctx_w, ef, "lm_head_proj.weight", D, D);
    model.lm_h0_b = load_1d(model.ctx_w, ef, "lm_head_proj.bias",   D);
    model.lm_h2_w = load_1d(model.ctx_w, ef, "lm_head_norm.weight", D);
    model.lm_h2_b = load_1d(model.ctx_w, ef, "lm_head_norm.bias",   D);
    model.lm_h3_w = load_2d(model.ctx_w, ef, "lm_head_out.weight",  D, V);
    model.lm_h3_b = load_1d(model.ctx_w, ef, "lm_head_out.bias",    V);

    esmf_close(ef);

    size_t used = ggml_used_mem(model.ctx_w);
    fprintf(stderr, "[esmc] weight context: %.1f MB  (%d tensors)\n",
            used / 1048576.0, g_tensors_loaded);
    fprintf(stderr, "[esmc] weights loaded (%d tensors)\n", g_tensors_loaded);
}

/* Norm helpers */

static ggml_tensor *layer_norm(ggml_context *ctx,
                                ggml_tensor *x,
                                ggml_tensor *w,
                                ggml_tensor *b,
                                float eps = 1e-5f)
{
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, w);
    if (b) x = ggml_add(ctx, x, b);
    return x;
}

/* Debug helper */

static void print_tensor_stats(const char *label, ggml_tensor *t)
{
    if (!t) { fprintf(stderr, "  %-30s  (not found)\n", label); return; }

    int64_t n = ggml_nelements(t);
    double sum = 0, sum2 = 0;
    float first5[5] = {};
    int show = (int)(n < 5 ? n : 5);

    for (int64_t i = 0; i < n; i++) {
        float v = ggml_get_f32_1d(t, i);
        sum  += v;
        sum2 += (double)v * v;
        if (i < show) first5[i] = v;
    }
    double mean = sum / n;
    double std  = sqrt(sum2/n - mean*mean);

    fprintf(stderr, "  %-30s  mean=%9.6f  std=%9.6f  first5=[", label, mean, std);
    for (int i = 0; i < show; i++)
        fprintf(stderr, "%s%.6f", i?", ":"", first5[i]);
    fprintf(stderr, "]\n");
}

/* Forward-pass graph builder */

static ggml_tensor *esmc_build_graph(
        esmc_model    &model,
        ggml_context  *ctx,
        ggml_cgraph   *gf,
        const int32_t *token_ids,
        int            seq_len)
{
    const esmc_hparams &hp = model.hparams;
    const int   L   = seq_len;
    const int   D   = hp.d_model;
    const int   NH  = hp.n_heads;
    const int   HD  = hp.head_dim;
    const int   F   = hp.ffn_hidden;
    const float rsf = 1.0f / hp.scaling_factor;
    const float attn_scale = 1.0f / sqrtf((float)HD);

    /* Input tokens */
    ggml_tensor *inp = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(inp, "input_tokens");
    if (inp->data) {
        memcpy(inp->data, token_ids, (size_t)L * sizeof(int32_t));
    }

    /* Position indices for RoPE */
    ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
    ggml_set_name(pos, "positions");
    if (pos->data) {
        int32_t *p = (int32_t *)pos->data;
        for (int i = 0; i < L; i++) p[i] = i;
    }

    /* Token embeddings: embed_w [D,V] × inp [L] → x [D,L] */
    ggml_tensor *x = ggml_get_rows(ctx, model.embed_w, inp);
    ggml_set_name(x, "embeddings");

    /* Transformer blocks */
    for (int li = 0; li < hp.n_layers; li++) {
        const esmc_layer &lw = model.layers[li];
        const bool dbg = (li == 0);

        /* ── Attention ── */

        /* Pre-LN */
        ggml_tensor *attn_in = layer_norm(ctx, x, lw.attn_ln_w, lw.attn_ln_b);
        if (dbg) ggml_set_name(attn_in, "attn_in");

        /* Fused QKV projection: qkv_w [D,3D] × attn_in [D,L] → [3D,L] */
        ggml_tensor *qkv = ggml_mul_mat(ctx, lw.qkv_w, attn_in);
        if (dbg) ggml_set_name(qkv, "qkv");

        /* Split Q, K, V — each [D, L] */
        const size_t elt    = ggml_element_size(qkv);
        const size_t row_nb = qkv->nb[1];
        ggml_tensor *Q = ggml_view_2d(ctx, qkv, D, L, row_nb, 0);
        ggml_tensor *K = ggml_view_2d(ctx, qkv, D, L, row_nb, elt*(size_t)D);
        ggml_tensor *V = ggml_view_2d(ctx, qkv, D, L, row_nb, elt*(size_t)D*2);
        if (dbg) { ggml_set_name(Q,"q"); ggml_set_name(K,"k"); ggml_set_name(V,"v"); }

        /* QK-Norm: LayerNorm, weight only, no bias */
        Q = layer_norm(ctx, Q, lw.q_ln_w, nullptr);
        K = layer_norm(ctx, K, lw.k_ln_w, nullptr);
        if (dbg) { ggml_set_name(Q,"q_qknorm"); ggml_set_name(K,"k_qknorm"); }

        /* Reshape to [HD, NH, L] */
        Q = ggml_reshape_3d(ctx, ggml_cont(ctx, Q), HD, NH, L);
        K = ggml_reshape_3d(ctx, ggml_cont(ctx, K), HD, NH, L);
        V = ggml_reshape_3d(ctx, ggml_cont(ctx, V), HD, NH, L);

        /* RoPE — applied after QK-Norm */
        Q = ggml_rope_ext(ctx, Q, pos, nullptr,
                          HD, hp.rope_type, hp.max_seq_len, hp.rope_base,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        K = ggml_rope_ext(ctx, K, pos, nullptr,
                          HD, hp.rope_type, hp.max_seq_len, hp.rope_base,
                          1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (dbg) { ggml_set_name(Q,"q_rope"); ggml_set_name(K,"k_rope"); }

        /* Permute Q and K to [HD, L, NH] */
        Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
        K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));

        /* Scores: mul_mat(K,Q) → [L, L, NH]  (dim-0=key, dim-1=query) */
        ggml_tensor *scores = ggml_mul_mat(ctx, K, Q);
        scores = ggml_scale(ctx, scores, attn_scale);
        scores = ggml_soft_max(ctx, scores);
        if (dbg) ggml_set_name(scores, "attn_weights");

        /* Permute V to [L, HD, NH] (ax0 = 1, ax1 = 2, ax2 = 0, ax3 = 3) */
        ggml_tensor *V_perm = ggml_cont(ctx, ggml_permute(ctx, V, 1, 2, 0, 3));

        /* Weighted sum: mul_mat(V_perm, scores) → [HD, L, NH] */
        ggml_tensor *attn_out = ggml_mul_mat(ctx, V_perm, scores);

        /* Merge heads: [HD, L, NH] → permute to [HD, NH, L] → reshape to [D, L] */
        attn_out = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
        attn_out = ggml_reshape_2d(ctx, attn_out, D, L);
        if (dbg) ggml_set_name(attn_out, "attn_ctx");

        /* Output projection */
        attn_out = ggml_mul_mat(ctx, lw.out_w, attn_out);
        if (dbg) ggml_set_name(attn_out, "attn_out");

        /* Residual: x += attn_out / scaling_factor */
        x = ggml_add(ctx, x, ggml_scale(ctx, attn_out, rsf));
        if (dbg) ggml_set_name(x, "x_attn");

        /* ── SwiGLU FFN ── */

        ggml_tensor *ffn_in = layer_norm(ctx, x, lw.ffn_ln_w, lw.ffn_ln_b);
        if (dbg) ggml_set_name(ffn_in, "ffn_in");

        /* Up-proj: fc1_w [D,2F] × ffn_in [D,L] → proj [2F,L] */
        ggml_tensor *proj = ggml_mul_mat(ctx, lw.fc1_w, ffn_in);
        if (dbg) ggml_set_name(proj, "ffn_proj");

        /* Split: gate = first F rows, value = second F rows */
        const size_t pelt = ggml_element_size(proj);
        const size_t pnb1 = proj->nb[1];
        ggml_tensor *gate  = ggml_view_2d(ctx, proj, F, L, pnb1, 0);
        ggml_tensor *value = ggml_view_2d(ctx, proj, F, L, pnb1, pelt*(size_t)F);
        if (dbg) { ggml_set_name(gate,"ffn_gate"); ggml_set_name(value,"ffn_value"); }

        /* SwiGLU: silu(gate) * value */
        ggml_tensor *swiglu = ggml_mul(ctx, ggml_silu(ctx, gate), value);
        if (dbg) ggml_set_name(swiglu, "swiglu");

        /* Down-proj: fc2_w [F,D] × swiglu [F,L] → [D,L] */
        ggml_tensor *ffn_out = ggml_mul_mat(ctx, lw.fc2_w, swiglu);
        if (dbg) ggml_set_name(ffn_out, "ffn_out");

        /* Residual: x += ffn_out / scaling_factor */
        x = ggml_add(ctx, x, ggml_scale(ctx, ffn_out, rsf));
        if (dbg) ggml_set_name(x, "x_block");
    }

    /* Final LayerNorm without bias */
    x = layer_norm(ctx, x, model.final_norm_w, nullptr);
    ggml_set_name(x, "final_norm");

    /* LM head: Linear(D→D) → GELU → LayerNorm → Linear(D→V)
     * confirmed from RegressionHead source in esm/layers/regression_head.py */
    ggml_tensor *h = ggml_mul_mat(ctx, model.lm_h0_w, x);
    h = ggml_add(ctx, h, model.lm_h0_b);
    h = ggml_gelu(ctx, h);
    h = layer_norm(ctx, h, model.lm_h2_w, model.lm_h2_b);
    ggml_tensor *logits = ggml_mul_mat(ctx, model.lm_h3_w, h);
    logits = ggml_add(ctx, logits, model.lm_h3_b);
    ggml_set_name(logits, "logits");

    ggml_build_forward_expand(gf, logits);
    return logits;
}

/* Forward pass runner */

static std::vector<float> esmc_forward(
        esmc_model &model, const std::vector<int32_t> &tokens,
        int n_threads = 4, bool debug = false)
{
    const int L  = (int)tokens.size();
    const int V  = model.hparams.vocab_size;
    const int NH = model.hparams.n_heads;
    const int NL = model.hparams.n_layers;
    const int F  = model.hparams.ffn_hidden;
    const int D  = model.hparams.d_model;

    // Create graph allocator
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());

    /* Only allocate metadata space inside the main context. */
    const size_t meta_mem = ggml_tensor_overhead() * 2048 + 2 * 1024 * 1024;
    struct ggml_init_params cp = { meta_mem, nullptr, true };
    ggml_context *ctx = ggml_init(cp);
    if (!ctx) {
        ggml_gallocr_free(galloc);
        throw std::runtime_error("ggml_init (compute) failed");
    }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, false);
    ggml_tensor *logits_t = esmc_build_graph(model, ctx, gf, tokens.data(), L);

    // Allocate the graph tensors using galloc
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_free(ctx);
        ggml_gallocr_free(galloc);
        throw std::runtime_error("ggml_gallocr_alloc_graph failed");
    }

    // Populate input tensors since memory has now been resolved by galloc
    ggml_tensor *inp = ggml_graph_get_tensor(gf, "input_tokens");
    ggml_tensor *pos = ggml_graph_get_tensor(gf, "positions");
    assert(inp && inp->data);
    assert(pos && pos->data);
    memcpy(inp->data, tokens.data(), (size_t)L * sizeof(int32_t));
    { int32_t *p = (int32_t *)pos->data; for (int i = 0; i < L; i++) p[i] = i; }

    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();

    if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        ggml_gallocr_free(galloc);
        throw std::runtime_error("ggml_graph_compute failed");
    }

    if (debug) {
        fprintf(stderr, "\n[validate] layer-0 intermediate activations:\n");
        static const char *tags[] = {
            "embeddings","attn_in","qkv","q","k","v",
            "q_qknorm","k_qknorm","q_rope","k_rope",
            "attn_weights","attn_ctx","attn_out","x_attn",
            "ffn_in","ffn_proj","ffn_gate","ffn_value","swiglu","ffn_out","x_block",
            nullptr
        };
        for (int i = 0; tags[i]; i++) {
            ggml_tensor *t = ggml_graph_get_tensor(gf, tags[i]);
            print_tensor_stats(tags[i], t);
        }
        fprintf(stderr, "\n");
    }

    /* Copy logits [V,L] → out[L,V] row-major */
    std::vector<float> out(L * V);
    for (int p = 0; p < L; p++)
        for (int v = 0; v < V; v++)
            out[p*V + v] = ggml_get_f32_nd(logits_t, v, p, 0, 0);

    ggml_free(ctx);
    ggml_gallocr_free(galloc);
    return out;
}

/* Output */

static void print_top(const float *row, int V, int k)
{
    float mx = row[0];
    for (int i = 1; i < V; i++) if (row[i] > mx) mx = row[i];
    std::vector<float> p(V); float s = 0;
    for (int i = 0; i < V; i++) { p[i] = expf(row[i]-mx); s += p[i]; }
    for (int i = 0; i < V; i++) p[i] /= s;
    std::vector<int> idx(V);
    for (int i = 0; i < V; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return p[a]>p[b]; });
    for (int i = 0; i < k && i < V; i++)
        printf("    %-8s %.4f\n", token_to_aa(idx[i]), p[idx[i]]);
}

/* main */

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <model.esmf> <sequence> [n_threads] [--debug]\n"
            "  X  = masked position (model predicts what fits)\n"
            "  --debug  prints layer-0 activation stats\n",
            argv[0]);
        return 1;
    }

    const char  *path      = argv[1];
    std::string  seq       = argv[2];
    int          n_threads = 4;
    bool         debug     = false;

    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--debug") debug = true;
        else n_threads = atoi(argv[i]);
    }

    fprintf(stderr, "[esmc] loading %s …\n", path);
    esmc_model model;
    try { esmc_load_weights(model, path); }
    catch (const std::exception &e) {
        fprintf(stderr, "load error: %s\n", e.what()); return 1;
    }

    auto tokens = tokenise(seq);
    const int L = (int)tokens.size();
    fprintf(stderr, "[esmc] %zu residues → %d tokens (with BOS/EOS)\n",
            seq.size(), L);

    if (L > model.hparams.max_seq_len) {
        fprintf(stderr, "error: sequence too long (%d > %d)\n",
                L, model.hparams.max_seq_len);
        return 1;
    }

    fprintf(stderr, "[esmc] forward pass (n_threads=%d) …\n", n_threads);
    std::vector<float> logits;
    try { logits = esmc_forward(model, tokens, n_threads, debug); }
    catch (const std::exception &e) {
        fprintf(stderr, "inference error: %s\n", e.what()); return 1;
    }

    const int V = model.hparams.vocab_size;
    printf("\nPer-position top-5 predictions:\n");
    printf("%-6s  %-8s\n", "pos", "input");
    printf("%.54s\n", "------------------------------------------------------");
    for (int pos = 1; pos < L-1; pos++) {
        printf("%-6d  %-8s\n", pos, token_to_aa(tokens[pos]));
        print_top(logits.data() + pos*V, V, 5);
    }

    ggml_free(model.ctx_w);
    return 0;
}