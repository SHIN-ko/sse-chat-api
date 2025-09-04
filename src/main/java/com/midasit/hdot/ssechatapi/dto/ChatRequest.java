package com.midasit.hdot.ssechatapi.dto;

public record ChatRequest(
        String system,
        String userPrompt
) {}