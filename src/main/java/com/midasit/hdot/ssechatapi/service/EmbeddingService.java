package com.midasit.hdot.ssechatapi.service;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.midasit.hdot.ssechatapi.config.AppEmbeddingProps;
import com.midasit.hdot.ssechatapi.config.NoopNDListTranslator;
import org.springframework.stereotype.Component;

@Component
public class EmbeddingService implements AutoCloseable {

    private final ZooModel<NDList, NDList> model;
    private final Predictor<NDList, NDList> predictor;
    private final HuggingFaceTokenizer tokenizer;

    public EmbeddingService(AppEmbeddingProps props) throws Exception {
        // 토크나이저 로드
        this.tokenizer = HuggingFaceTokenizer.newInstance(props.getModel());

        // 모델 로드 (PyTorch)
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .optEngine("PyTorch")
                .setTypes(NDList.class, NDList.class)
                .optModelUrls("djl://ai.djl.huggingface.pytorch/" + props.getModel())
                .optTranslator(new NoopNDListTranslator())
                .build();

        this.model = criteria.loadModel();
        this.predictor = model.newPredictor();
    }

    /** 단일 문장 임베딩 */
    public float[] embedOne(String text) throws TranslateException {
        NDManager manager = NDManager.newBaseManager();
        try {
            var enc = tokenizer.encode(text);
            long[] ids = enc.getIds();
            long[] mask = enc.getAttentionMask();

            NDArray inputIds = manager.create(ids).reshape(new Shape(1, ids.length)).toType(DataType.INT64, false);
            NDArray attnMask = manager.create(mask).reshape(new Shape(1, mask.length)).toType(DataType.INT64, false);

            NDList inputs = new NDList();
            inputs.add(inputIds);
            inputs.add(attnMask);

            NDList out = predictor.predict(inputs); // [last_hidden_state] (1, seq_len, hidden)
            NDArray tokenEmb = out.get(0);

            NDArray maskFloat = attnMask.toType(DataType.FLOAT32, false);
            NDArray sumEmb = tokenEmb.mul(maskFloat.expandDims(-1)).sum(new int[]{1});      // (1, hidden)
            NDArray len = maskFloat.sum(new int[]{1}).clip(1e-9f, Float.MAX_VALUE);        // (1,)
            NDArray mean = sumEmb.div(len.expandDims(-1));                                  // (1, hidden)

            float[] vec = mean.toFloatArray();
            // L2 normalize (cosine 안정화)
            float norm = 0f;
            for (float v : vec) norm += v * v;
            norm = (float)Math.sqrt(norm) + 1e-10f;
            for (int i=0;i<vec.length;i++) vec[i] /= norm;

            return vec;
        } finally {
            manager.close();
        }
    }

    @Override public void close() { predictor.close(); model.close(); }
}