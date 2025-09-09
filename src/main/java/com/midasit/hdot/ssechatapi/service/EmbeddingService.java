package com.midasit.hdot.ssechatapi.service;

import ai.djl.ModelException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

@Service
public class EmbeddingService {

    @Value("${app.embedding.onnx.model-dir}")
    private String modelDir;

    @Value("${app.embedding.onnx.max-length:256}")
    private int maxLen;

    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;

    @PostConstruct
    public void init() throws Exception {
        Path base = Path.of(modelDir).toAbsolutePath().normalize();

        // 디버그: 실제 폴더 내용
        System.out.println("[Embedding] modelDir=" + base);
        if (java.nio.file.Files.isDirectory(base)) {
            try (var s = java.nio.file.Files.list(base)) {
                s.forEach(p -> System.out.println("  - " + p.getFileName()));
            }
        }

        // 1) tokenizer.json 경로 결정 (루트 → onnx → 재귀 탐색)
        Path tokFile = null;
        Path cand1 = base.resolve("tokenizer.json");
        Path cand2 = base.resolve("onnx").resolve("tokenizer.json");
        if (java.nio.file.Files.isRegularFile(cand1)) tokFile = cand1;
        else if (java.nio.file.Files.isRegularFile(cand2)) tokFile = cand2;
        else tokFile = findFirst(base, "tokenizer.json", 5);

        if (tokFile == null) {
            throw new IllegalStateException("tokenizer.json not found under: " + base);
        }

        // ★ DJL 0.25.0에서 확실히 있는 오버로드: Path 버전 사용
        this.tokenizer = ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
                .newInstance(tokFile.toRealPath());   // ← String 말고 Path!

        // 2) ONNX 파일도 찾기
        Path onnxFile = null;
        Path c1 = base.resolve("model.onnx");
        Path c2 = base.resolve("onnx").resolve("model.onnx");
        if (java.nio.file.Files.isRegularFile(c1)) onnxFile = c1;
        else if (java.nio.file.Files.isRegularFile(c2)) onnxFile = c2;
        else onnxFile = findFirstByExt(base, ".onnx", 5);
        if (onnxFile == null) {
            throw new IllegalStateException(".onnx model file not found under: " + base);
        }

        // 3) ONNX 세션 로드
        this.env = ai.onnxruntime.OrtEnvironment.getEnvironment();
        var so = new ai.onnxruntime.OrtSession.SessionOptions();
        so.setOptimizationLevel(ai.onnxruntime.OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session = env.createSession(onnxFile.toRealPath().toString(), so);

        System.out.println("[Embedding] tokenizer.json = " + tokFile.toRealPath());
        System.out.println("[Embedding] model.onnx     = " + onnxFile.toRealPath());
    }

    // 유틸
    private Path findFirst(Path root, String fileName, int maxDepth) throws java.io.IOException {
        try (var s = java.nio.file.Files.walk(root, maxDepth)) {
            return s.filter(java.nio.file.Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().equals(fileName))
                    .findFirst().orElse(null);
        }
    }
    private Path findFirstByExt(Path root, String ext, int maxDepth) throws java.io.IOException {
        try (var s = java.nio.file.Files.walk(root, maxDepth)) {
            return s.filter(java.nio.file.Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase().endsWith(ext))
                    .findFirst().orElse(null);
        }
    }

    @PreDestroy
    public void close() {
        try { if (session != null) session.close(); } catch (Exception ignore) {}
        try { if (env != null) env.close(); } catch (Exception ignore) {}
        try { if (tokenizer != null) tokenizer.close(); } catch (Exception ignore) {}
    }


    public float[] embedOne(String text) {
        try {
            // [1] 토크나이즈 & 마스크 생성
            Encoding enc = tokenizer.encode(text);
            long[] ids   = padOrTruncate(enc.getIds(), maxLen);
            long[] attn  = padAttention(enc.getAttentionMask(), maxLen);
            long[] typeIds = new long[maxLen]; // 모두 0

            // [2] ONNX 입력 텐서 (batch = 1)
            long[] shape = new long[]{1, maxLen};
            try (OnnxTensor inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape);
                 OnnxTensor attention = OnnxTensor.createTensor(env, LongBuffer.wrap(attn), shape);
                 OnnxTensor tokenType = OnnxTensor.createTensor(env, LongBuffer.wrap(typeIds), shape)) {

                // 입력 이름 자동 매핑
                Map<String, OnnxTensor> inputs = new HashMap<>();
                Set<String> inNames = session.getInputNames();
                for (String name : inNames) {
                    String key = name.toLowerCase(Locale.ROOT);
                    if (key.contains("input_ids")) inputs.put(name, inputIds);
                    else if (key.contains("attention_mask")) inputs.put(name, attention);
                    else if (key.contains("token_type_ids") || key.contains("token_type_id")) inputs.put(name, tokenType);
                }
                if (!inputs.values().contains(inputIds) || !inputs.values().contains(attention)) {
                    throw new IllegalStateException("Unexpected ONNX input names: " + inNames);
                }

                // [3] 추론
                try (OrtSession.Result result = session.run(inputs)) {
                    // 출력 우선순위: sentence_embedding/embeddings → last_hidden_state → 첫 번째 텐서
                    List<String> outOrder = new ArrayList<>(session.getOutputNames());
                    outOrder.sort((a, b) -> scoreOutName(b) - scoreOutName(a)); // 높은 점수 우선

                    for (String outName : outOrder) {
                        var opt = result.get(outName);
                        if (opt.isPresent() && opt.get() instanceof OnnxTensor t) {
                            Object val = t.getValue();
                            long[] outShape = t.getInfo().getShape(); // 예: [1, 384] or [1, seq, 384]

                            // 디버그: 필요 시 1회 찍고 비활성화 가능
                            // System.out.println("OUT name=" + outName + ", shape=" + Arrays.toString(outShape) + ", class=" + val.getClass());

                            // (1, d)
                            if (val instanceof float[][] vec2) {
                                return vec2[0];
                            }
                            // (1, seq, d)
                            if (val instanceof float[][][] vec3) {
                                float[][] tokenEmb = vec3[0];              // [seq, dim]
                                float[] pooled = meanPool(tokenEmb, attn); // attention mask로 평균
                                return l2norm(pooled);
                            }
                            // 드물게 (d) 단독
                            if (val instanceof float[] vec1) {
                                return l2norm(vec1);
                            }
                        }
                    }
                    throw new IllegalStateException("No usable tensor output found. Output names=" + session.getOutputNames());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Embedding failed: " + e.getMessage(), e);
        }
    }

    /** 출력 이름 스코어링: 문장 임베딩명을 우선 선택 */
    private static int scoreOutName(String name) {
        String k = name.toLowerCase(Locale.ROOT);
        if (k.contains("sentence_embedding")) return 100;
        if (k.contains("embeddings")) return 90;
        if (k.contains("last_hidden_state")) return 80;
        return 10; // 기타는 낮게
    }

    /** mean pooling (attention mask가 1인 토큰만 평균) */
    private static float[] meanPool(float[][] tokenEmbeddings, long[] attnMask) {
        int seq = tokenEmbeddings.length;
        int dim = tokenEmbeddings[0].length;
        float[] out = new float[dim];
        long valid = 0;

        for (int i = 0; i < seq; i++) {
            boolean use = (attnMask == null) || (i < attnMask.length && attnMask[i] != 0);
            if (use) {
                valid++;
                float[] ti = tokenEmbeddings[i];
                for (int d = 0; d < dim; d++) out[d] += ti[d];
            }
        }
        if (valid == 0) valid = seq;
        float inv = 1.0f / (float) valid;
        for (int d = 0; d < dim; d++) out[d] *= inv;
        return out;
    }

    /** L2 정규화 (e5 등 권장) */
    private static float[] l2norm(float[] v) {
        double s = 0.0;
        for (float x : v) s += x * x;
        double n = Math.sqrt(s);
        if (n == 0) return v;
        for (int i = 0; i < v.length; i++) v[i] = (float)(v[i] / n);
        return v;
    }

    // ---- tokenizer util (이미 가지고 있다면 기존 것 사용)
    private long[] padOrTruncate(long[] ids, int maxLen) {
        long[] out = new long[maxLen];
        int n = Math.min(ids.length, maxLen);
        System.arraycopy(ids, 0, out, 0, n);
        if (n < maxLen) Arrays.fill(out, n, maxLen, 0); // pad token id가 0이 아니면 교체
        return out;
    }

    private long[] padAttention(long[] mask, int maxLen) {
        long[] out = new long[maxLen];
        int n = Math.min(mask.length, maxLen);
        System.arraycopy(mask, 0, out, 0, n);
        if (n < maxLen) Arrays.fill(out, n, maxLen, 0);
        return out;
    }
}