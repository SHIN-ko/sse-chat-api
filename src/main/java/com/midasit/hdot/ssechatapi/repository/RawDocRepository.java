package com.midasit.hdot.ssechatapi.repository;

import com.midasit.hdot.ssechatapi.dto.DocIngestRq;
import org.springframework.stereotype.Repository;
import software.amazon.awssdk.services.dynamodb.DynamoDbClient;
import software.amazon.awssdk.services.dynamodb.model.AttributeValue;
import software.amazon.awssdk.services.dynamodb.model.PutItemRequest;

import java.util.Map;

@Repository
public class RawDocRepository {

    private final DynamoDbClient ddb;
    private final String tableName = "rag_raw_docs";

    public RawDocRepository(DynamoDbClient ddb) { this.ddb = ddb; }

    public void save(DocIngestRq rq) {
        PutItemRequest put = PutItemRequest.builder()
                .tableName(tableName)
                .item(Map.of(
                        "doc_id", AttributeValue.fromS(rq.docId()),
                        "title",  AttributeValue.fromS(rq.title() == null ? "" : rq.title()),
                        "text",   AttributeValue.fromS(rq.text()),
                        "metadata", AttributeValue.fromS(rq.metadata() == null ? "{}" : rq.metadata().toString())
                )).build();
        ddb.putItem(put);
    }
}