import re
from langchain.schema import Document


# regex used for spltting
MAIN_RE = re.compile(r"RULE[ \u00A0]+(?P<main_id>\d{1,3})[ \u00A0]+(?P<main_name>[A-Z-'´’–/”“]{3,}+(?:[ \u00A0][A-Z-'´’–/”“]+)*)")
SUB_RE = re.compile(r"(?P<sub_id>\d{1,3}\.\d{1,2})[.\u00A0 ]*[ \u00A0]+(?P<sub_name>[A-Z-'´’–/”“]{2,}+(?:[ \u00A0][A-Z-'´’–/”“0-9]+)*)")



def merge_docs(docs: list[Document], source: str) -> list[Document]:
    """Merges data split by the dataloader back into one document.
    
    Adds specified metadata and follows the schema of the Langchain Document.

    Args:
        docs (list[Document]): A list of split Document objects
        source (str): Source of the document to be used for metadata

    Returns:
        list[Document]: A list containing a single merged Document object
    """
    parts = []
    # sticth docs back together
    for i, d in enumerate(docs, start=1):
        parts.append(f"\n\n<<<PAGE {i}>>>\n{d.page_content.strip()}")
    merged_content = "".join(parts)
    
    # create the Document structure with page_content and metadata
    merged_doc = [
        Document(
            page_content=merged_content,
            metadata={
                "source": source,
            }
        )
    ]
    return merged_doc





def normalize_ocr(text: str) -> str:
    """Normalize common OCR artifacts in extracted PDF text.

    This function fixes spacing issues that occur when 
    PDF text extraction splits words incorrectly.
    
    For example, "PENAL TY" → "PENALTY" or "AL TERNATE" → "ALTERNATE".

    Args:
        text (str): The input text to be normalized

    Returns:
        str: The normalized text with correct spacing
    """
    # collapse weird spaces (regular + non-breaking)
    text = re.sub(r"[ \u00A0]+", " ", text)

    # fix error for 'penalty' which somtimes is 'penal ty' etc
    text = re.sub(r"\bPENAL\s*TY\b", "PENALTY", text)
    text = re.sub(r"\bPENAL\s*TIES\b", "PENALTIES", text)
    text = re.sub(r"\bAL\s*TERNATE\b", "ALTERNATE", text)
    
    return text




# adds rule id and name to metadata
def add_rule_metadata(m: re.Match, prefix: str) -> dict:
    """Creates metadata from a regex match for main rules and sub rules

    Args:
        m (re.Match): Regex match object
        prefix (str): Whether it is a Main or Sub rule.

    Returns:
        dict: A dictionary with keys and values of rule_name and rule_id for main and sub rules
    """
    metadata = {
        f"{prefix}_rule": m.group(0),
        f"{prefix}_rule_id": m.group(1),
        f"{prefix}_rule_name": m.group(2),
    }
    return metadata




# split on rules and subrules
def slice_on_regex(docs: list[Document], pattern: re.Pattern, prefix: str) -> list[Document]:
    """Split LangChain Documents into rule-level chunks based on a regex pattern.
    
    Each regex match marks the start of a new main- or sub-rule.

    Args:
        docs (list[Document]): Input documents to be sliced.
        pattern (re.Pattern): Compiled regular expression used to identify rule headers.
        prefix (str): Metadata prefix to use (e.g., "main" or "sub").

    Returns:
        list[Document]: A list of new Document objects. Each object is either a 
        full main-rule or sub-rule depending on {prefix}. It's metadata now contains
        fields for {prefix_rule_id} and {prefix_rule_name}.
    """
    out = []

    for d in docs:
        text = normalize_ocr(d.page_content).lstrip()
        matches = list(pattern.finditer(text))
        
        # returns original document if no matches.
        if not matches:
            out.append(d)
            continue
        
        # For each regex match, slice from its start up to the next match (or end of text),
        # so each chunk corresponds to one rule section with its content. 
        for i, m in enumerate(matches):
            if i + 1 < len(matches):
                end = matches[i+1].start()  
            else:
                end = len(text)
            chunk = text[m.start():end]
            
            # get new rule metadata and update metadata for the chunk
            rule_metadata = add_rule_metadata(m, prefix)
            new_meta = d.metadata.copy()
            new_meta.update(rule_metadata)
            
            out.append(Document(page_content=chunk, metadata=new_meta))

    return out

