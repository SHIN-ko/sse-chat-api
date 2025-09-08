package com.midasit.hdot.ssechatapi.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

@Component
@ConfigurationProperties(prefix = "app.embedding")
public class AppEmbeddingProps {
    private String model;
    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }
}