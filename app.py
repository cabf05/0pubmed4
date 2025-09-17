import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from itertools import chain
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="PubMed Hot Topics HF", layout="wide")
st.title("🔍 PubMed Hot Topics Explorer (Hugging Face NER)")

query = st.text_area(
    "PubMed Search Query", 
    value='("Endocrinology" OR "Diabetes") AND 2024/10/01:2025/06/28[Date - Publication]'
)
max_results = st.number_input("Max number of articles", min_value=10, max_value=500, value=50, step=10)
hf_token = st.text_input("Hugging Face API Token", type="password")
hf_model = "d4data/biobert-cased-finetuned-ner"

generic_terms = set([
    "study", "patient", "patients", "trial", "results", "effect", "effects",
    "group", "clinical", "analysis", "evaluation", "treatment", "data"
])

if st.button("🔎 Run Analysis"):
    with st.spinner("Fetching articles..."):
        # Buscar PMIDs
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "retmax": str(max_results), "retmode": "json", "term": query}
        r = requests.get(esearch_url, params=params)
        id_list = r.json()["esearchresult"].get("idlist", [])

        efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"}
        response = requests.get(efetch_url, params=params, timeout=20)

        records = []
        try:
            root = ET.fromstring(response.content)
            articles = root.findall(".//PubmedArticle")
            for article in articles:
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle", "")
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                journal = article.findtext(".//Journal/Title", "")
                date = article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "N/A"
                records.append({"PMID": pmid, "Title": title, "Link": link, "Journal": journal, "Date": date})
        except:
            st.error("Failed to parse PubMed XML.")

        df = pd.DataFrame(records)

        # Hugging Face NER
        def get_entities(text):
            if not hf_token or not text.strip():
                return []
            headers = {"Authorization": f"Bearer {hf_token}"}
            payload = {"inputs": text}
            try:
                r = requests.post(f"https://api-inference.huggingface.co/models/{hf_model}",
                                  headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    ents = r.json()
                    return [e['word'] for e in ents if 'word' in e and e['word'].lower() not in generic_terms]
            except:
                return []
            return []

        st.info("Running Hugging Face NER on article titles...")
        df['entities'] = df['Title'].apply(get_entities)

        # N-grams
        def get_ngrams(entity_lists, n=2):
            ngrams = []
            for entities in entity_lists:
                tokens = [re.sub(r'\W+', '', e).lower() for e in entities if e]
                ngrams.extend([" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
            return ngrams

        bigrams = get_ngrams(df['entities'], n=2)
        trigrams = get_ngrams(df['entities'], n=3)

        # Wordcloud function
        def plot_wordcloud(words, title):
            if words:
                word_freq = Counter(words)
                wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                plt.figure(figsize=(15,6))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.subheader(title)
                st.pyplot(plt)
                plt.close()
            else:
                st.info(f"No entities found to generate {title}.")

        plot_wordcloud(list(chain.from_iterable(df['entities'])), "Wordcloud - Single Words")
        plot_wordcloud(bigrams, "Wordcloud - Bigrams")
        plot_wordcloud(trigrams, "Wordcloud - Trigrams")

        # DataFrame e CSV
        st.subheader("Article Table")
        st.dataframe(df[['PMID','Title','Journal','Date','entities']], use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", data=csv, file_name="pubmed_entities.csv", mime="text/csv")
