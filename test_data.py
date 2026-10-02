import pandas as pd

df = pd.read_csv("global_supply_chain.csv")


def total_trade_value():
    return df["Trade Value"].sum()


def top_supplier():
    result = df.groupby("Supplier")["Trade Value"].sum()
    return result.idxmax(), result.max()


def top_importer():
    result = df.groupby("importer")["Trade Value"].sum()
    return result.idxmax(), result.max()


def trade_by_year():
    return df.groupby("Year")["Trade Value"].sum().sort_index()


def supplier_totals():
    return df.groupby("Supplier")["Trade Value"].sum().sort_values(ascending=False)


def importer_totals():
    return df.groupby("importer")["Trade Value"].sum().sort_values(ascending=False)


print("Total Trade Value:", total_trade_value())

supplier, value = top_supplier()
print("Top Supplier:", supplier)
print("Trade Value:", value)

importer, value = top_importer()
print("Top Importer:", importer)
print("Trade Value:", value)

print("\nTrade by Year:")
print(trade_by_year())

print("\nSupplier Totals:")
print(supplier_totals())

print("\nImporter Totals:")
print(importer_totals())