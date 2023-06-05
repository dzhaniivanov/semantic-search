import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAI } from "langchain/llms/openai";
import { loadQAStuffChain } from "langchain/chains";
import { Document } from "langchain/document";
import { timeout } from "./config";

export const createPineconeIndex = async (
  client,
  indexName,
  vectorDimension
) => {
  console.log(`checking ${indexName}...`);
  const existingIndexes = await client.listIndexes();

  if (!existingIndexes.includes(indexName)) {
    console.log(`creating ${indexName}...`);

    await client.createIndex({
      createRequest: {
        name: indexName,
        dimension: vectorDimension,
        metric: "cosine",
      },
    });
    console.log(`creating index...please wait for it to finish initializing`);

    await new Promise((resolve) => setTimeout(resolve, timeout));
  } else {
    console.log(`${indexName} already exists`);
  }
};

export const updatePinecone = async (client, indexName, docs) => {
  const index = client.Index(indexName);
  console.log(`pinecone index retrieved : ${indexName}`);

  for (const doc of docs) {
    console.log(`processing document : ${doc.metadata.source}`);
    const txtPath = doc.metadata.source;
    const text = doc.pageContent;

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    });
    console.log("splitting text into chunks...");
    const chunks = await textSplitter.createDocuments([text]);
    console.log(
      `calling OpenAi embedding endpoint documents with ${chunks.length} text chunks...`
    );
    const embedingArrays = await new OpenAIEmbeddings().embedDocuments(
      chunks.map((chunk) => chunk.pageContent.replace(/\n/g, ""))
    );
    console.log(
      `creating ${chunks.length} vectors array with id,values and metadata...`
    );

    const batchSize = 100;
    let batch: any = [];
    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx];
      const vector = {
        id: `${txtPath}_${idx}`,
        values: embedingArrays[idx],
        metadata: {
          ...chunk.metadata,
          loc: JSON.stringify(chunk.metadata.loc),
          pageContent: chunk.pageContent,
          txtPath,
        },
      };
      batch = [...batch, vector];

      if (batch.length === batchSize || idx == chunks.length - 1) {
        await index.upsert({
          upsertRequest: {
            vectors: batch,
          },
        });
        batch = [];
      }
    }
  }
};
