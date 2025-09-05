package com.midasit.hdot.ssechatapi.controller;

import com.midasit.hdot.ssechatapi.dto.ChatRequest;
import com.midasit.hdot.ssechatapi.dto.ChatResponse;
import com.midasit.hdot.ssechatapi.service.ModelClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Duration;


@RestController
@RequestMapping("/api/chat")
public class ChatStreamController {

    private final ModelClient modelClient;

    @Value("${app.sse.heartbeat-seconds:20}")
    private long heartbeatSec;

    public ChatStreamController(ModelClient modelClient) {
        this.modelClient = modelClient;
    }

    /** 1) 비스트리밍: 최종 텍스트 한 번에 반환 */
    @PostMapping(produces = MediaType.APPLICATION_JSON_VALUE, consumes = MediaType.APPLICATION_JSON_VALUE)
    public Mono<ChatResponse> chat(@RequestBody ChatRequest req) {
        String system = nvl(req.system());
        String up = nvl(req.userPrompt());
        return modelClient.complete(system, up)
                .defaultIfEmpty("")
                .map(ChatResponse::new);
    }

    /** 2-A) 스트리밍(SSE, GET) — 여러 파라미터 이름(userPrompt|user|prompt|q) 수용 */
    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<String>> streamGet(
            @RequestParam(value = "system", required = false, defaultValue = "") String system,
            @RequestParam(value = "userPrompt", required = false) String userPrompt,
            @RequestParam(value = "user", required = false) String user,
            @RequestParam(value = "prompt", required = false) String prompt,
            @RequestParam(value = "q", required = false) String q
    ) {
        String up = firstNonBlank(userPrompt, user, prompt, q);
        return sse(system, up);
    }

    /** 2-B) 스트리밍(SSE, POST) — fetch(POST) 스트림 */
    @PostMapping(value = "/stream",
            produces = MediaType.TEXT_EVENT_STREAM_VALUE,
            consumes = MediaType.APPLICATION_JSON_VALUE)
    public Flux<ServerSentEvent<String>> streamPost(@RequestBody ChatRequest req) {
        return sse(req.system(), req.userPrompt());
    }

    /** 공통 SSE 구현 (빈 스트림/지연 시 폴백 포함) */
    private Flux<ServerSentEvent<String>> sse(String system, String userPrompt) {
        String sys = nvl(system);
        String up  = nvl(userPrompt);

        if (isBlank(up)) {
            // 프롬프트 없으면 에러 한 번 보내고 종료
            return Flux.concat(
                    Flux.just(ServerSentEvent.<String>builder("[error] missing userPrompt").event("message").build()),
                    Flux.just(ServerSentEvent.<String>builder("").event("done").build())
            );
        }

        // 업스트림 스트림 → 첫 토큰 7초 지연 시/빈 스트림 시 논스트리밍으로 폴백
        var dataFlux = modelClient.streamDeltas(sys, up)
                .filter(s -> s != null && !s.isBlank())
                .timeout(Duration.ofSeconds(7), modelClient.complete(sys, up).flux())  // 첫 토큰 타임아웃 폴백
                .switchIfEmpty(Mono.defer(() -> modelClient.complete(sys, up)).flux()) // 빈 스트림 폴백
                .map(delta -> ServerSentEvent.<String>builder(delta).event("message").build())
                .onErrorResume(ex -> Flux.just(
                        ServerSentEvent.<String>builder("[error] " + ex.getMessage()).event("message").build()
                ));

        // 하트비트: dataFlux 종료 시 자동 종료
        var heartbeat = Flux.interval(Duration.ofSeconds(heartbeatSec))
                .map(i -> ServerSentEvent.<String>builder("").comment("ping").build())
                .takeUntilOther(dataFlux.ignoreElements().then());

        return Flux.concat(
                Flux.merge(dataFlux, heartbeat),
                Flux.just(ServerSentEvent.<String>builder("").event("done").build())
        );
    }
    private static String firstNonBlank(String... ss) { for (String s: ss) if (!isBlank(s)) return s; return null; }
    private static boolean isBlank(String s) { return s == null || s.trim().isEmpty(); }
    private static String nvl(String s) { return s == null ? "" : s; }
}