package com.midasit.hdot.ssechatapi.service;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;


@Service
public class RerankerService {

    @Value("${app.reranker.onnx.model}")
    private String modelDir;

    @Value("${app.reranker.onnx.max-length:256}")
    private int maxLen;

    private OrtEnvironment env;
    private OrtSession session;
    private HuggingFaceTokenizer tokenizer;

    @PostConstruct
    public void init() throws Exception {
        Path base = Path.of(modelDir).toAbsolutePath().normalize();

        // 디버그: 폴더 내용 찍기
        System.out.println("[Reranker] modelDir=" + base);
        if (Files.isDirectory(base)) {
            try (var s = Files.list(base)) {
                s.forEach(p -> System.out.println("  - " + p.getFileName()));
            }
        }

        // tokenizer.json 찾기
        Path tokFile = null;
        Path cand1 = base.resolve("tokenizer.json");
        if (Files.isRegularFile(cand1)) tokFile = cand1;
        else tokFile = findFirst(base, "tokenizer.json", 5);
        if (tokFile == null) {
            throw new IllegalStateException("tokenizer.json not found under: " + base);
        }

        // model.onnx 찾기
        Path onnxFile = null;
        Path c1 = base.resolve("model.onnx");
        if (Files.isRegularFile(c1)) onnxFile = c1;
        else onnxFile = findFirstByExt(base, ".onnx", 5);
        if (onnxFile == null) {
            throw new IllegalStateException("model.onnx not found under: " + base);
        }

        // 토크나이저 로드
        this.tokenizer = ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
                .newInstance(tokFile.toRealPath());

        // ONNX 세션 로드
        this.env = OrtEnvironment.getEnvironment();
        var so = new OrtSession.SessionOptions();
        so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session = env.createSession(onnxFile.toRealPath().toString(), so);

        System.out.println("[Reranker] tokenizer.json = " + tokFile.toRealPath());
        System.out.println("[Reranker] model.onnx     = " + onnxFile.toRealPath());
        System.out.println("[Reranker] inputs         = " + this.session.getInputNames());
        System.out.println("[Reranker] outputs        = " + this.session.getOutputNames());
    }

    @PreDestroy
    public void close() {
        try { if (session != null) session.close(); } catch (Exception ignore) {}
        try { if (env != null) env.close(); } catch (Exception ignore) {}
        try { if (tokenizer != null) tokenizer.close(); } catch (Exception ignore) {}
    }

    /** (query, doc) 쌍을 점수화: 보통 logit을 반환. 필요시 시그모이드 적용해서 0~1 확률로 바꿔도 됨. */
    public float score(String query, String doc) {
        try {
            // 1) pair 인코딩 (query + doc)
            Encoding enc = tokenizer.encode(query, doc);
            long[] ids   = padOrTrunc(enc.getIds(), maxLen);
            long[] attn  = padOrTrunc(enc.getAttentionMask(), maxLen);
            // 일부 모델은 type ids 필요없음 → 안전하게 0으로 채운 벡터 준비
            long[] typeIds = (enc.getTypeIds() != null && !Arrays.equals(enc.getTypeIds(), new long[0]))
                    ? padOrTrTrunc(enc.getTypeIds(), maxLen)
                    : zeros(maxLen);

            // 2) 입력 텐서 (batch=1)
            long[] shape = new long[]{1, maxLen};
            try (OnnxTensor inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape);
                 OnnxTensor attention = OnnxTensor.createTensor(env, LongBuffer.wrap(attn), shape);
                 OnnxTensor tokenType = OnnxTensor.createTensor(env, LongBuffer.wrap(typeIds), shape)) {

                // 3) 입력 이름 매핑 (모델마다 이름 다를 수 있음)
                Map<String, OnnxTensor> inputs = new HashMap<>();
                Set<String> inNames = session.getInputNames();
                for (String name : inNames) {
                    String k = name.toLowerCase(Locale.ROOT);
                    if (k.contains("input_ids")) inputs.put(name, inputIds);
                    else if (k.contains("attention_mask")) inputs.put(name, attention);
                    else if (k.contains("token_type_ids") || k.contains("token_type_id")) inputs.put(name, tokenType);
                }

                // 최소한 input_ids/attention_mask는 들어가야 함
                if (!inputs.values().contains(inputIds) || !inputs.values().contains(attention)) {
                    throw new IllegalStateException("Unexpected ONNX input names: " + inNames);
                }

                // 4) 추론
                try (OrtSession.Result result = session.run(inputs)) {
                    // 일반적으로 cross-encoder는 (1, 1) 또는 (1,) float 출력 (logit)
                    for (String outName : session.getOutputNames()) {
                        var opt = result.get(outName);
                        if (opt.isPresent() && opt.get() instanceof OnnxTensor t) {
                            Object val = t.getValue();
                            if (val instanceof float[][] m && m.length > 0 && m[0].length > 0) {
                                return m[0][0]; // (1,1)
                            } else if (val instanceof float[] v && v.length > 0) {
                                return v[0];    // (1,)
                            }
                        }
                    }
                    // 못 찾으면 첫 텐서에서 best-effort
                    var first = result.get(0);
                    if (first instanceof OnnxTensor t) {
                        Object val = t.getValue();
                        if (val instanceof float[][] m && m.length > 0 && m[0].length > 0) return m[0][0];
                        if (val instanceof float[] v && v.length > 0) return v[0];
                    }
                    throw new IllegalStateException("No usable scalar output in reranker. outputs=" + session.getOutputNames());
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Reranker failed: " + e.getMessage(), e);
        }
    }

    // --------- 유틸 ---------
    private static long[] padOrTrunc(long[] arr, int maxLen) {
        long[] out = new long[maxLen];
        int n = Math.min(arr.length, maxLen);
        System.arraycopy(arr, 0, out, 0, n);
        if (n < maxLen) Arrays.fill(out, n, maxLen, 0);
        return out;
    }
    private static long[] padOrTrTrunc(long[] list, int maxLen) {
        long[] out = new long[maxLen];
        int n = Math.min(list.length, maxLen);
        for (int i = 0; i < n; i++) out[i] = list[i];
        if (n < maxLen) Arrays.fill(out, n, maxLen, 0);
        return out;
    }
    private static long[] zeros(int n) {
        return new long[n]; // 0으로 채워짐
    }
    private static Path findFirst(Path root, String fileName, int maxDepth) throws java.io.IOException {
        try (var s = java.nio.file.Files.walk(root, maxDepth)) {
            return s.filter(java.nio.file.Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().equals(fileName))
                    .findFirst().orElse(null);
        }
    }
    private static Path findFirstByExt(Path root, String ext, int maxDepth) throws java.io.IOException {
        try (var s = java.nio.file.Files.walk(root, maxDepth)) {
            return s.filter(java.nio.file.Files::isRegularFile)
                    .filter(p -> p.getFileName().toString().toLowerCase().endsWith(ext))
                    .findFirst().orElse(null);
        }
    }
}