// com.example.rag.config.DynamoConfig
package com.midasit.hdot.ssechatapi.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;

@Configuration
public class DynamoConfig {

    @Bean
    public DynamoDbClient dynamoDbClient() {
        return DynamoDbClient.builder()
                .region(Region.AP_NORTHEAST_2)                 // ✅ 리전만 지정
                .credentialsProvider(DefaultCredentialsProvider.create()) // ✅ 기본 크레덴셜 체인
                .build();
    }
}
