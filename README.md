# Global Supply Chain AI Assistant

A chat-based data-analysis application built with **Python**, **Pandas**, and **Streamlit** to explore global trade data. The application uses predefined, rule-based question patterns to retrieve results from the dataset.

---

## Project Overview

The **Global Supply Chain AI Assistant** is an interactive application designed to simplify the exploration of global trade data.

Users can enter questions through a chat interface and retrieve information from a structured global trade dataset. Instead of manually filtering and aggregating data, users can ask questions about:

- Total trade value
- Suppliers
- Importers
- Yearly trade activity
- Supplier trade totals
- Importer trade totals
- Selected country-to-country trade relationships

The application matches supported questions to predefined patterns and performs the corresponding data analysis using Python and Pandas. The Streamlit application does not use a language model or an external AI API.

---

## Project Background

This project began as a **Power BI dashboard** built from international trade data collected from the **UN Comtrade** database. After building the dashboard, I extended the same dataset into a chat-based interface using Python, Pandas, and Streamlit, so the same trade metrics can be queried by typing questions instead of using dashboard filters.

**Project progression:**

```text
UN Comtrade Data Collection
          ↓
Data Cleaning and Preparation
          ↓
Power BI Dashboard
          ↓
Python + Pandas Analysis
          ↓
Streamlit Chat Interface
```

---

## Objectives

- Provide a simple chat interface for exploring global trade data.
- Analyze trade values across suppliers and importing countries.
- Identify major suppliers and importers based on trade value.
- Examine trade activity across different years.
- Simplify access to aggregated trade information.
- Apply Python-based data analysis to a real-world supply chain dataset.

---

## Data Source

**UN Comtrade – International Trade Database**

The dataset (`global_supply_chain.csv`) contains 2,538 records of historical trade data covering the years 2019–2023, with 4 supplier countries and 3 importing countries.

| Column | Description |
|---|---|
| Year | Year of the trade record |
| importer | Importing country |
| Supplier | Supplier country |
| Trade Flow | Type of trade flow |
| Trade Value | Value of the trade |

The dataset is stored in CSV format and processed using Pandas.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Power BI | Initial trade-data dashboard |
| Python | Application development and data processing |
| Pandas | Data filtering, grouping, aggregation, and analysis |
| Streamlit | Interactive web application and chat interface |
| CSV | Structured trade-data storage |

---

## Key Features

### Total Trade Analysis
- Calculates the overall trade value across the dataset.
- Returns the aggregated trade value through the chat interface.

### Supplier Analysis
- Calculates total trade values for individual suppliers.
- Identifies the largest and smallest suppliers.
- Displays supplier-wise trade totals.

### Importer Analysis
- Calculates total trade values for individual importers.
- Identifies the largest and smallest importers.
- Displays importer-wise trade totals.

### Yearly Trade Analysis
- Groups trade values by year.
- Calculates total trade value for each year.
- Displays yearly trade results in tabular format.

### Country-to-Country Trade Analysis
- Supports analysis of Germany's trade with China (Germany as importer, China as supplier).
- Calculates the corresponding total trade value for this relationship.

### Chat Interface
- Allows users to submit questions through a chat interface.
- Displays questions and responses in conversational format.
- Maintains chat history during the application session.
- Provides a **New Chat** option.
- Displays recent questions in the sidebar.

---

## Example Queries

| Example Query | Result |
|---|---|
| What is the total trade value? | Overall trade value |
| Which supplier is the biggest? | Largest supplier |
| Which supplier is the smallest? | Smallest supplier |
| Which importer is the biggest? | Largest importer |
| Which importer is the smallest? | Smallest importer |
| Show supplier totals | Supplier-wise trade values |
| Show importer totals | Importer-wise trade values |
| Show me trade by year | Yearly trade values |
| How much did China trade? | China's total supplier trade value |
| What is Germany's trade value with China? | Germany's trade value with China |

---

## Data Processing and Analysis

The application uses Pandas to perform the main analytical operations:

- Loading the trade dataset into a DataFrame
- Grouping records by supplier
- Grouping records by importer
- Calculating aggregate trade values
- Identifying maximum and minimum trade values
- Grouping trade values by year
- Filtering records for specific country relationships
- Sorting analytical results
- Preparing results for display in the Streamlit interface

---

## Application Workflow

```text
Trade Dataset
     ↓
CSV Data Loading
     ↓
Pandas Data Processing
     ↓
User Question
     ↓
Question Pattern Matching
     ↓
Relevant Data Analysis
     ↓
Result Generation
     ↓
Chat Display
```

### Workflow Steps

1. **Data Loading** – The application loads the global trade dataset from a CSV file.
2. **Question Input** – The user enters a question through the Streamlit chat interface.
3. **Question Processing** – The application converts the question to lowercase and checks it against the supported query patterns.
4. **Data Analysis** – The corresponding Pandas operation is performed on the dataset.
5. **Result Generation** – The calculated value or table is prepared for display.
6. **Response Display** – The result is displayed in the chat interface.

---

## Application Screenshots

### Main Interface

![Main Interface](Global%20Supply%20Chain%20AI%20Assistant.png)

*Main interface showing the chat workspace, recent questions, and chat input.*

### Dark Theme

![Dark Theme](Global%20Supply%20Chain%20AI%20Assistant%20Dark%20Theme.png)

*The application interface in dark theme.*

### Trade Analysis Result

![Trade by Year Table](Global%20Supply%20Chain%20AI%20Assistant%20Table.png)

*Example of the yearly trade-value table generated from the dataset.*

---

## Project Structure

```text
global-supply-chain-ai-assistant/
│
├── app.py                  # Streamlit chat application
├── chatbot.py              # Standalone API testing script
├── test_data.py            # Script that prints sample analyses from the dataset
├── global_supply_chain.csv # Trade dataset
├── README.md
├── .gitignore
├── LICENSE
├── Global Supply Chain AI Assistant.png
├── Global Supply Chain AI Assistant Dark Theme.png
└── Global Supply Chain AI Assistant Table.png
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/ammu-analytics/global-supply-chain-ai-assistant.git
```

### 2. Navigate to the Project Directory

```bash
cd global-supply-chain-ai-assistant
```

### 3. Install Dependencies

```bash
pip install streamlit pandas
```

### 4. Run the Application

```bash
streamlit run app.py
```

### 5. Open the Application

After running the command, open the local Streamlit URL shown in the terminal, typically:

```text
http://localhost:8501
```

### Optional: Run the Data Check Script

```bash
python test_data.py
```

---

## Limitations

- The application is rule-based and supports predefined question patterns only.
- Questions outside the implemented patterns may not be recognized.
- The analysis depends on the information available in the underlying dataset.
- Country-to-country analysis is currently implemented only for Germany (importer) and China (supplier).
- The application is intended primarily for data exploration and educational purposes.

---

## Future Enhancements

Potential improvements (not currently implemented):

- Integrating a language model for more flexible natural-language understanding
- Expanding support for country-to-country trade queries
- Adding interactive charts and visualizations
- Adding commodity-level trade analysis
- Improving follow-up question handling
- Expanding the underlying trade dataset
- Adding additional supply-chain indicators

---

## Conclusion

The Global Supply Chain AI Assistant demonstrates how a trade-data analysis project can move from a Power BI dashboard to an interactive, rule-based chat application built with Python, Pandas, and Streamlit.

The project provides a simple way to retrieve and analyze information related to:

- Trade values
- Suppliers
- Importers
- Yearly trade activity
- Country-to-country trade relationships

Through this project, I gained practical experience in data collection, dashboard development, Python programming, data aggregation, analytical operations, and the development of interactive data-driven applications.

---

## Author

**Lakshaya S**

## Project Category

- Data Analytics
- Business Intelligence
- Conversational Data Applications
- Python Development
- Supply Chain Analytics
