from langchain_core.documents import Document
import json
from pathlib import Path
path=Path("data/transaction.json")

# load json
def load_transaction(path):
    with path.open(mode="r",encoding="utf-8") as f:
        data=json.load(f)
        return data

transactions=load_transaction(path)
def json_to_doc(transaction):
    docs=[]
    for data in transactions:
        date=data['date']
        name=data["customer"]
        product=data["product"]
        amount=data["amount"]
    
        sentence=f"On {date}, {name} purchased a {product} for ₹{amount}"
        metadata={
            "id":data["id"],"source":path.name
        }
        doc=Document(
            page_content=sentence,
            metadata=metadata
        )
        docs.append(doc)
    return docs

docs=json_to_doc(transactions)
print("Document created Successfully")
