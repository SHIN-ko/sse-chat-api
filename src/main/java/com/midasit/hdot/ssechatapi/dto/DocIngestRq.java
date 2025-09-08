package com.midasit.hdot.ssechatapi.dto;

import java.util.Map;

public record DocIngestRq(
        String docId,
        String title,
        String text,
        Map<String, Object> metadata
) {}