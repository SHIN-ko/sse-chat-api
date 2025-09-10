package com.midasit.hdot.ssechatapi.dto;

import java.util.Map;

public record OsHit(
        String id,
        float score,
        String title,
        String text,
        Map<String, Object> metadata
) {}