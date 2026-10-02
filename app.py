import streamlit as st
import pandas as pd

# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Global Supply Chain AI",
    page_icon="🌍",
    layout="wide"
)

# ==================================================
# LOAD CSV
# ==================================================

df = pd.read_csv("global_supply_chain.csv")

# ==================================================
# CHAT MEMORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🌍 Supply Chain AI")

    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    st.subheader("📜 Recent Questions")

    if len(st.session_state.messages) == 0:

        st.caption("No questions yet.")

    else:

        for message in st.session_state.messages:

            if message["role"] == "user":

                st.write("💬 " + message["content"])


# ==================================================
# MAIN PAGE
# ==================================================

st.title("🌍 Global Supply Chain AI Assistant")

st.caption(
    "Ask questions about your Global Supply Chain data."
)

st.divider()


# ==================================================
# DISPLAY OLD CHAT
# ==================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):

            if message["type"] == "text":

                st.markdown(message["content"])

            elif message["type"] == "table":

                st.markdown(message["content"])

                st.dataframe(
                    message["data"],
                    use_container_width=True
                )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask a question about your supply-chain data..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================

if question:

    # Save user's question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Convert everything to lowercase
    # This makes uppercase/lowercase irrelevant.
    q = question.lower().strip()

    answer = ""
    answer_type = "text"
    answer_data = None


    # ==================================================
    # TOTAL TRADE
    # ==================================================

    if (
        "total trade" in q
        or "total trade value" in q
        or "overall trade" in q
        or "total value" in q
    ):

        total = df["Trade Value"].sum()

        answer = (
            f"💰 **Total trade value:** "
            f"{total:,.2f}"
        )


    # ==================================================
    # BIGGEST / HIGHEST SUPPLIER
    # ==================================================

    elif (
        "supplier" in q
        and (
            "biggest" in q
            or "largest" in q
            or "highest" in q
            or "most" in q
            or "top" in q
        )
    ):

        result = (
            df.groupby("Supplier")["Trade Value"]
            .sum()
            .sort_values(ascending=False)
        )

        supplier = result.index[0]
        value = result.iloc[0]

        answer = (
            f"🏆 **{supplier}** is the biggest supplier "
            f"with a total trade value of **{value:,.2f}**."
        )


    # ==================================================
    # SMALLEST / LOWEST SUPPLIER
    # ==================================================

    elif (
        "supplier" in q
        and (
            "smallest" in q
            or "lowest" in q
            or "least" in q
        )
    ):

        result = (
            df.groupby("Supplier")["Trade Value"]
            .sum()
            .sort_values()
        )

        supplier = result.index[0]
        value = result.iloc[0]

        answer = (
            f"📉 **{supplier}** is the smallest supplier "
            f"with a total trade value of **{value:,.2f}**."
        )


    # ==================================================
    # BIGGEST / HIGHEST IMPORTER
    # ==================================================

    elif (
        "importer" in q
        and (
            "biggest" in q
            or "largest" in q
            or "highest" in q
            or "most" in q
            or "top" in q
        )
    ):

        result = (
            df.groupby("importer")["Trade Value"]
            .sum()
            .sort_values(ascending=False)
        )

        importer = result.index[0]
        value = result.iloc[0]

        answer = (
            f"🏆 **{importer}** is the biggest importer "
            f"with a total trade value of **{value:,.2f}**."
        )


    # ==================================================
    # SMALLEST / LOWEST IMPORTER
    # ==================================================

    elif (
        "importer" in q
        and (
            "smallest" in q
            or "lowest" in q
            or "least" in q
        )
    ):

        result = (
            df.groupby("importer")["Trade Value"]
            .sum()
            .sort_values()
        )

        importer = result.index[0]
        value = result.iloc[0]

        answer = (
            f"📉 **{importer}** is the smallest importer "
            f"with a total trade value of **{value:,.2f}**."
        )


    # ==================================================
    # SUPPLIER TOTALS
    # ==================================================

    elif (
        "supplier totals" in q
        or "supplier total" in q
        or "supplier trade" in q
        or "show suppliers" in q
        or "show supplier totals" in q
    ):

        result = (
            df.groupby("Supplier")["Trade Value"]
            .sum()
            .sort_values(ascending=False)
        )

        answer = "🌍 **Supplier trade totals:**"

        answer_type = "table"

        answer_data = result


    # ==================================================
    # IMPORTER TOTALS
    # ==================================================

    elif (
        "importer totals" in q
        or "importer total" in q
        or "importer trade" in q
        or "show importers" in q
        or "show importer totals" in q
    ):

        result = (
            df.groupby("importer")["Trade Value"]
            .sum()
            .sort_values(ascending=False)
        )

        answer = "🌎 **Importer trade totals:**"

        answer_type = "table"

        answer_data = result


    # ==================================================
    # TRADE BY YEAR
    # ==================================================

    elif (
        "trade by year" in q
        or "trade value by year" in q
        or "yearly trade" in q
        or "trade each year" in q
        or "trade every year" in q
    ):

        result = (
            df.groupby("Year")["Trade Value"]
            .sum()
            .sort_index()
        )

        answer = "📅 **Trade value by year:**"

        answer_type = "table"

        answer_data = result


    # ==================================================
    # GERMANY + CHINA
    # ==================================================

    elif (
        "germany" in q
        and "china" in q
    ):

        result = df[
            (df["importer"].str.lower() == "germany")
            &
            (df["Supplier"].str.lower() == "china")
        ]

        total = result["Trade Value"].sum()

        answer = (
            f"🇩🇪 Germany's trade value with China is "
            f"**{total:,.2f}**."
        )


    # ==================================================
    # INDIVIDUAL SUPPLIER
    # ==================================================

    else:

        found_supplier = None

        # Get supplier names directly from the CSV.
        suppliers = df["Supplier"].dropna().unique()

        for supplier in suppliers:

            if supplier.lower() in q:

                found_supplier = supplier

                break


        if found_supplier:

            total = df[
                df["Supplier"].str.lower()
                == found_supplier.lower()
            ]["Trade Value"].sum()

            answer = (
                f"📦 **{found_supplier}**'s total trade "
                f"value is **{total:,.2f}**."
            )


        else:

            answer = (
                "I don't understand that question yet. "
                "Try asking about total trade, suppliers, "
                "importers, years, or a specific supplier."
            )


# ==================================================
# SAVE AI RESPONSE
# ==================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "type": answer_type,
            "data": answer_data
        }
    )

    st.rerun()