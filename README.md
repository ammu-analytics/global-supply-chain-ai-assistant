# Global Supply Chain AI Assistant

A conversational data-analysis application built with **Python**, **Pandas**, and **Streamlit** to explore global trade data through a chat-based interface.

---

## Project Overview

The **Global Supply Chain AI Assistant** is an interactive application designed to simplify the exploration of global trade data.

Users can enter questions through a conversational interface and retrieve relevant information from a structured global trade dataset. Instead of manually filtering and aggregating large datasets, users can ask questions about:

- Total trade value
- Suppliers
- Importers
- Yearly trade activity
- Supplier trade totals
- Importer trade totals
- Selected country-to-country trade relationships

The application processes supported questions and performs the corresponding data analysis using Python and Pandas.

---

## Objectives

- Provide a simple conversational interface for exploring global trade data.
- Analyze trade values across suppliers and importing countries.
- Identify major suppliers and importers based on trade value.
- Examine trade activity across different years.
- Simplify access to aggregated trade information.
- Apply Python-based data analysis to a real-world supply chain dataset.

---

## Data Source

**UN Comtrade – International Trade Database**

The project uses international trade data containing information related to:

- Importing countries
- Supplier countries
- Trade values
- Years
- Importer–supplier relationships

The dataset is stored in CSV format and processed using Pandas.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development and data processing |
| Pandas | Data filtering, grouping, aggregation, and analysis |
| Streamlit | Interactive web application and chat interface |
| CSV | Structured trade-data storage |

---

## Key Features

### Total Trade Analysis
- Calculates the overall trade value across the dataset.
- Returns the aggregated trade value through the conversational interface.

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
- Supports analysis of the Germany–China trade relationship.
- Calculates the corresponding total trade value for the supported relationship.

### Conversational Interface
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
| What is Germany's trade value with China? | Germany–China trade value |

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
Conversational Display
```

### Workflow Steps

1. **Data Loading** – The application loads the global trade dataset from a CSV file.
2. **Question Input** – The user enters a question through the Streamlit chat interface.
3. **Question Processing** – The application converts the question to lowercase and identifies the relevant supported query pattern.
4. **Data Analysis** – The corresponding Pandas operation is performed on the dataset.
5. **Result Generation** – The calculated value or table is prepared for display.
6. **Response Display** – The result is displayed through the conversational interface.

---

## Application Screenshots

### Main Interface – Dark Theme

![Main Interface – Dark Theme](Global%20Supply%20Chain%20AI%20Assistant%20Dark%20Theme.png)

*Dark theme interface showing the conversational workspace, recent questions, and chat input.*

### Trade Value by Year – Light Theme

![Trade Value by Year – Light Theme](Global%20Supply%20Chain%20AI%20Assistant%20Table.png)

*Light theme interface showing yearly trade-value analysis from 2019 to 2023.*

---

## Project Structure

```text
global-supply-chain-ai-assistant/
│
├── app.py
├── chatbot.py
├── test_data.py
├── global_supply_chain.csv
├── requirements.txt
├── README.md
├── .gitignore
├── LICENSE
│
└── images/
    ├── main-interface.png
    └── trade-by-year.png
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
pip install -r requirements.txt
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

---

## Limitations

- The application supports predefined question patterns only.
- Questions outside the implemented patterns may not be recognized.
- The analysis depends on the information available in the underlying dataset.
- Country-to-country analysis is currently implemented for supported relationships only.
- The application is intended primarily for data exploration and educational purposes.

---

## Future Enhancements

- Integrating a language model for more flexible natural-language understanding
- Expanding support for country-to-country trade queries
- Adding interactive charts and visualizations
- Adding commodity-level trade analysis
- Improving conversational follow-up questions
- Expanding the underlying trade dataset
- Adding additional supply-chain indicators
- Improving automated analytical insights
- Deploying the application to a public web platform

---

## Conclusion

The Global Supply Chain AI Assistant demonstrates how Python, Pandas, and Streamlit can be combined to create an interactive conversational application for exploring international trade data.

The project provides a simple way to retrieve and analyze information related to:

- Trade values
- Suppliers
- Importers
- Yearly trade activity
- Country-to-country trade relationships

Through this project, I gained practical experience in Python programming, data processing, data aggregation, analytical operations, and the development of interactive data-driven applications.

---

## Author

**Lakshaya S**

## Project Category

- Data Analytics
- Conversational Data Applications
- Python Development
- Supply Chain Analytics
- Data-Driven Applications
