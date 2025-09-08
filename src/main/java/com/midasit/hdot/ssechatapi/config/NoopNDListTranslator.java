package com.midasit.hdot.ssechatapi.config;

import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class NoopNDListTranslator implements Translator<NDList, NDList> {
    @Override public NDList processInput(TranslatorContext ctx, NDList input) { return input; }
    @Override public NDList processOutput(TranslatorContext ctx, NDList output) { return output; }
}