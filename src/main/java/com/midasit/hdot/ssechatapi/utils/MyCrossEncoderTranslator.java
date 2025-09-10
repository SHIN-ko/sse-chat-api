package com.midasit.hdot.ssechatapi.utils;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.ndarray.*;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Pair;

import java.nio.file.Path;

public class MyCrossEncoderTranslator implements NoBatchifyTranslator<Pair<String, String>, Float> {

        private HuggingFaceTokenizer tokenizer;
        private final int maxLen;

    public MyCrossEncoderTranslator() { this(256); }
    public MyCrossEncoderTranslator(int maxLen) { this.maxLen = maxLen; }

        @Override
        public void prepare(TranslatorContext ctx) throws Exception {
        Path modelPath = ctx.getModel().getModelPath();
        tokenizer = HuggingFaceTokenizer.newInstance(modelPath); // models/ms-marco-... 폴더의 tokenizer.json 사용
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Pair<String, String> input) {
        String query = input.getKey();
        String doc   = input.getValue();

        Encoding enc = tokenizer.encode(query, doc);

        long[] ids   = padOrTrunc(enc.getIds(), maxLen);
        long[] mask  = padOrTrunc(enc.getAttentionMask(), maxLen);
        // enc.getTypeIds()가 비어있거나 null일 수 있음 (RoBERTa/DistilBERT 계열)
        long[] typeIdsList = enc.getTypeIds();
        boolean hasTypeIds = typeIdsList != null;
        long[] types = hasTypeIds ? padOrTrunc(typeIdsList, maxLen) : null;

        NDArray inputIds = ctx.getNDManager().create(ids).reshape(1, maxLen);
        inputIds.setName("input_ids");

        NDArray attentionMask = ctx.getNDManager().create(mask).reshape(1, maxLen);
        attentionMask.setName("attention_mask");

        NDList in = new NDList(inputIds, attentionMask);

        if (hasTypeIds) {
            NDArray tokenTypeIds = ctx.getNDManager().create(types).reshape(1, maxLen);
            tokenTypeIds.setName("token_type_ids");
            in.add(tokenTypeIds);
        }

        return in;
    }

        @Override
        public Float processOutput(TranslatorContext ctx, NDList list) {
        // 보통 [1] 또는 [1,1] 형태 logit
        return list.singletonOrThrow().toFloatArray()[0];
        // 필요시 sigmoid 적용: (float)(1.0 / (1.0 + Math.exp(-logit)));
    }

        private static long[] padOrTrunc(long[] src, int maxLen) {
        long[] out = new long[maxLen];
        int n = Math.min(src.length, maxLen);
        for (int i = 0; i < n; i++) out[i] = src[i];
        return out; // 나머지는 0 패딩
    }
}
