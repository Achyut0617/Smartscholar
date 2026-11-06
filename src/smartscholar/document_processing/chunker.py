#import hashlib
#import json
#from langchain_text_splitters import RecursiveCharacterTextSplitter

#class Chunker:
#    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
#        self.chunk_size = chunk_size
#        self.chunk_overlap = chunk_overlap
#        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

#    def _generate_metadata(self, chunk: str, document_id: str, chunk_order: int) -> dict:
#        token_count = len(chunk.split())  # Token count (word count)
#        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()  # Chunk hash

#        return {
#            "document_id": document_id,
#            "chunk_order": chunk_order,
#            "text": chunk,
#            "token_count": token_count,
#            "chunk_hash": chunk_hash
#        }

#    def chunk_text(self, text: str, document_id: str):
#        chunks = self.text_splitter.split_text(text)
#        chunked_data = [
#            self._generate_metadata(chunk, document_id, i) for i, chunk in enumerate(chunks)
#        ]
#        return chunked_data

#    def save_to_json(self, chunked_data, filename="chunked_output.json"):
#        with open(filename, "w", encoding="utf-8") as f:
#            json.dump(chunked_data, f, indent=4, ensure_ascii=False)

# Example usage
# if __name__ == "__main__":
#     with open("sampletxt.txt", 'r', encoding="utf-8") as file:
#         text = file.read()

#     chunker = Chunker(chunk_size=500, chunk_overlap=50)
#     document_id = "sampletxt.txt"

#     chunks_with_metadata = chunker.chunk_text(text, document_id)

#     # Print chunks
#     for chunk in chunks_with_metadata:
#         print(chunk, "\n", "-" * 40)

#     # Save to JSON
#     chunker.save_to_json(chunks_with_metadata)


# // new approach 

import hashlib
import json
from pathlib import Path
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Token-aware splitting using tiktoken
        self.text_splitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        )

    def _generate_metadata(
        self,
        chunk: str,
        document_id: str,
        chunk_order: int,
        source: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> dict:

        chunk_hash = hashlib.sha256(
            chunk.encode("utf-8")
        ).hexdigest()

        return {
            "document_id": document_id,
            "chunk_id": f"{document_id}_{chunk_order}",
            "chunk_order": chunk_order,
            "text": chunk,
            "token_count": len(
                self.text_splitter._tokenizer.encode(chunk)
            ),
            "character_count": len(chunk),
            "chunk_hash": chunk_hash,
            "source": source,
            "page_number": page_number
        }

    def chunk_text(
        self,
        text: str,
        document_id: str,
        source: Optional[str] = None
    ) -> list:

        if not text or not text.strip():
            return []

        chunks = self.text_splitter.split_text(text)

        return [
            self._generate_metadata(
                chunk=chunk,
                document_id=document_id,
                chunk_order=i,
                source=source
            )
            for i, chunk in enumerate(chunks)
        ]

    def save_to_json(
        self,
        chunked_data: list,
        filename: str = "chunked_output.json"
    ):

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(
                chunked_data,
                f,
                indent=4,
                ensure_ascii=False
            )



