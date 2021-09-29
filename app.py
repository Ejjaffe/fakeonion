import streamlit as st
from fakeonion import Generator, Comparisons

# some app values
prev_epoch = 65
N_DEFAULT_CANDIDATE_TOKENS = 5
DEFAULT_NOMODEL = "__No text generated yet!__"
generated_text = DEFAULT_NOMODEL
comparisons_out = []

with open("app_explanation_md.txt", "r") as f:
    explanation = "\n\n".join(f.readlines())

generator = Generator(model_dir="models", init_epoch=prev_epoch)
comparisons = Comparisons(data_file="all_onion_headlines.txt")

# text at the top
st.title("Fake Onion Headlines with DistilGPT2")
st.text('Elias Jaffe, 2021')
st.markdown("## Interactive App")
# the sidebar
sidebar_epoch = st.sidebar.select_slider(
    "Epoch:", options=list(range(50, 90, 5)), value=prev_epoch
)
sidebar_genmode = st.sidebar.checkbox(
    "Make text more random", help="Doubles the number of candidate tokens."
)

# the text generation controls -------------------------------
with st.form("gen_form"):
    # input and submit
    seed = st.text_input("Generate a headline starting with this text:")
    n_comparisons = st.slider(
        "Compare to this number of the closest matching real headlines:",
        min_value=1,
        value=3,
        max_value=5,
    )
    do_generate = st.form_submit_button("Generate Me A Headline!")

    # on submit
    if do_generate:

        # if the epoch has changed, load that model
        if sidebar_epoch != prev_epoch:
            prev_epoch = sidebar_epoch
            generator.set_epoch(prev_epoch)

        # double the number of candidate tokens if requested
        cand_token_mult = 1 + int(sidebar_genmode)
        n_cand_token = cand_token_mult * N_DEFAULT_CANDIDATE_TOKENS

        # ask the generator to make some text
        generated_text = generator.generate_clean(seed=seed, token_cands=n_cand_token)

        # compare it to the real data
        comparisons_out = comparisons.compare(generated_text, n_comparisons)

st.subheader("Generated Headline:")
st.write(generated_text)

st.subheader("Most Similar Real Data:")
for comparison in comparisons_out:
    st.write(f"* {comparison['text']} | _Match: {comparison['score']}%_")
if len(comparisons_out) == 0:
    st.write(DEFAULT_NOMODEL)

st.markdown(explanation)
