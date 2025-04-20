# ShieldAgent 

## Automatic Policy Extraction Pipeline


Based on MCP, this agent extracts policies and rules from **PDF**, **HTML**, and **TXT** documents and produce structured risk categories governed by corresponding rule definitions.

> `deep policy exploration` feature: Designed for automatically exploring comprehensive policies among lengthy PDF documents or distributed HTML websites via a priority-based search algorithm that explores document subsections in-depth.

## ✨ Features

- 📚 Supports multiple document types (PDF, HTML, TXT)
- 🔍 Deep exploration of document sections and links via tree-search
- ⚙️ Prioritizes sections based on likelihood of containing target policies
- 🔄 Parallel processing support for faster extraction
- 📊 Comprehensive output including document trees and visualizations
- 🏷️ Automatic rules extraction and risk categorization

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/BillChan226/ShieldAgent.git
cd ShieldAgent

# Install required dependencies
pip install -r requirements.txt
```

## 📋 Usage

```bash
python policy_extractor_async.py [OPTIONS]
```

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--document-path` | `-d` | Path to document file (PDF, TXT) or URL (HTML) |
| `--organization` | `-org` | Name of the organization whose policies are being extracted |
| `--input-type` | `-t` | `pdf` | Type of input document (`pdf`, `html`, or `txt`) |


### Optional Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--organization-description` | `-org-desc` | `""` | Description of the organization |
| `--target-subject` | `-ts` | `User` | Target subject of the policies (e.g., "User", "Customer") |
| `--initial-page-range` | `-ipr` | `1-5` | Initial page range to extract from PDF |
| `--deep-policy` | `-dp` | `False` | Flag to automatically explore policy documents in-depth |
| `--output-dir` | `-o` | `./output/deep_policy` | Directory to save output files |
| `--debug` | | `False` | Enable debug mode |
| `--user-request` | `-u` | `""` | Specific user request to guide the policy extraction |
| `--async-num` | `-a` | `1` | Number of policy extraction tasks to run in parallel (1-3) |
| `--extract-rules` | `-er` | `False` | Extract concrete rules from policies after extraction |

## 📄 Input Types

The tool supports three input types:

### 📑 PDF Files (`-t pdf`)
- Supports page ranges with `-ipr` (e.g., "1-5" for pages 1 through 5)
- Automatically extracts text from specified pages
- Useful for official policy documents, terms of service, etc.

### 🌐 HTML/Web Pages (`-t html`)
- Extracts content from web pages
- Can follow links for deep policy extraction when `-dp` is specified
- Ideal for online policy repositories and terms of service pages

### 📝 Text Files (`-t txt`)
- Directly processes plain text files
- Simple and fastest processing option
- Good for pre-extracted content or manually curated policy text

## 🚀 Example Use Cases

### 1️⃣ Extract Policies from a Website (with Deep Exploration)

This example extracts policies from Reddit's rules page, follows links to related pages, and extracts rules targeting users:

```bash
python policy_extractor_async.py \
  -d https://redditinc.com/policies/reddit-rules \
  -t html \
  -org "Reddit" \
  -u "Focus on rules that target users and customers" \
  -dp \
  -er
```

### 2️⃣ Extract Policies from a PDF (Specific Pages)

This example extracts policies from the EU AI Act PDF, focusing on pages 2-8:

```bash
python policy_extractor_async.py \
  -d ./policy_docs/eu_ai_act_art5.pdf \
  -t pdf \
  -org "EU AI ACT" \
  -ipr "2-8" \
  -er
```

### 3️⃣ Extract Policies from a Text File

This example extracts policies from a text file containing EU AI Act rules:

```bash
python policy_extractor_async.py \
  -d ./eu_ai_act_rules/art_5.txt \
  -t txt \
  -org "EU AI ACT" \
  -er
```

### 4️⃣ Process Multiple Sections in Parallel

For large documents, you can enable parallel processing of sections:

```bash
python policy_extractor_async.py \
  -d large_policy_document.pdf \
  -t pdf \
  -org "Large Organization" \
  -a 3 \
  -er
```

## 📂 Output Files

The tool generates several output files in the specified output directory:

| File | Description |
|------|-------------|
| `{organization}_all_extracted_policies.json` | All extracted policies |
| `{organization}_all_extracted_rules.json` | Concrete rules extracted from policies |
| `{organization}_risk_categories.json` | Categorized rules with risk categories |
| `{organization}_document_tree.json` | Hierarchical representation of document sections with policies |
| `{organization}_extraction_report.json` | Detailed report of the extraction process |
| `{organization}_policy_rule_mapping.json` | Mapping between policies and rules |

## 🔍 Advanced Usage

### Custom Organization Description

Provide a detailed description of the organization to improve extraction quality:

```bash
python risk_extraction_doc/policy_extractor_async.py \
  -d ./company_policies.pdf \
  -org "Acme Corp" \
  -org-desc "A multinational technology company specializing in AI services" \
  -er
```

### Targeted Subject Extraction

Focus extraction on specific subjects within policies:

```bash
python policy_extractor_async.py \
  -d platform_guidelines.pdf \
  -org "Social Platform" \
  -ts "Content Creator" \
  -u "Focus on monetization policies" \
  -er
```

## ⚠️ Troubleshooting

- If you encounter errors with PDF processing, ensure you have all dependencies installed
- For HTML extraction issues, check if the website allows scraping
- For large documents, consider using page ranges (`-ipr`) to process specific sections
- Enable debug mode (`--debug`) for detailed logging information

## 🔄 Processing Flow

1. **Document Loading**: The tool loads the document based on input type
2. **Content Extraction**: Text is extracted from the document
3. **Section Analysis**: The system analyzes sections for policy content
4. **Policy Extraction**: Policies are identified and extracted
5. **Deep Exploration**: If enabled, the system explores linked sections
6. **Rules Extraction**: Concrete rules are extracted from policies
7. **Risk Categorization**: Rules are categorized by risk type
8. **Output Generation**: Results are saved to JSON files

## 📊 Example Output Structure

Policy structure:
```json
{
  "policy_id": "p123",
  "organization": "Example Org",
  "policy_text": "Users must not share personal information of others without consent.",
  "policy_source": "Privacy Policy, Section 3.2",
  "extraction_timestamp": "2023-10-25T14:30:00"
}
```

Rule structure:
```json
{
  "rule_id": "r456",
  "source_policy_ids": ["p123"],
  "rule_text": "Do not share another user's personal information without their explicit consent",
  "risk_category": "Privacy Violation",
  "severity": "High"
}
```

## 📚 Resources

- [EU AI Act](https://digital-strategy.ec.europa.eu/en/policies/european-approach-artificial-intelligence)
- [Policy Extraction Techniques Paper](https://example.org/paper)
- [Related Work on Policy Analysis](https://example.org/work)

## 📄 License

[Your license information here]
