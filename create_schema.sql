CREATE TABLE bazaar_item (
    id TEXT PRIMARY KEY
);

CREATE TABLE bazaar_item_history (
    item_id TEXT REFERENCES bazaar_item(id),
    timestamp TIMESTAMPTZ NOT NULL,
    buy REAL,
    sell REAL,
    buy_volume BIGINT,
    sell_volume BIGINT,
    max_buy REAL,
    min_buy REAL,
    max_sell REAL,
    min_sell REAL,
    buy_moving_week BIGINT,
    sell_moving_week BIGINT,
    PRIMARY KEY (item_id, timestamp)
);