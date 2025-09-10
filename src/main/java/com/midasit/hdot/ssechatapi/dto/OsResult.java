package com.midasit.hdot.ssechatapi.dto;

import java.util.List;

public record OsResult(
        List<OsHit> hits,
        int tookMs
) {}