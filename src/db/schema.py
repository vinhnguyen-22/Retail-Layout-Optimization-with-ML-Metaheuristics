SCHEMA = {
    "sku": {
        "Sku": "int32",
        "VendorId": "int32",
        "Item_id": "int8",
        "Dept": "int16",
        "Sdept": "int16",
        "Class": "int16",
        "Sclass": "int16",
        "IsPrivateLabel": "bool",
        "Created": "Int32",
    },
    "store_adjust": {"StoreId": "int32", "StoreIdBSC": "int32"},
    "transactions": {
        "StoreId": "int32",
        "Date": "int32",
        "Time": "int32",
        "Sku": "int32",
        "Sales": "float32",
        "CustomerId": "Int64",
        "MergedId": "int64",
    },
}
