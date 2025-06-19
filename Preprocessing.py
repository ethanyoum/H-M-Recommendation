# Statistics of Missing values
missing_values_customers = customers.isnull().sum()
missing_values_articles = articles.isnull().sum()
missing_values_transactions = transactions.isnull().sum()

missing_values_summary = pd.DataFrame({
    'Dataset': (
        ['Customers'] * len(missing_values_customers) +
        ['Articles'] * len(missing_values_articles) +
        ['Transactions'] * len(missing_values_transactions)
    ),
    'Column': (
        list(missing_values_customers.index) +
        list(missing_values_articles.index) +
        list(missing_values_transactions.index)
    ),
    'Missing Values': (
        list(missing_values_customers.values) +
        list(missing_values_articles.values) +
        list(missing_values_transactions.values)
    ),
    'Missing Percentage': (
        list((missing_values_customers / len(customers) * 100).values) +
        list((missing_values_articles / len(articles) * 100).values) +
        list((missing_values_transactions / len(transactions) * 100).values)
    )
})

missing_values_summary

# Remove FN and Active columns in the customers data
customers = customers.drop(columns = ['FN', 'Active'])
customers.head(5)

# Remove rest of the columns except for article_id and product_code for articles dataset
columns_keep = ['article_id', 'product_code']
articles = articles[columns_keep]

### Check if club_member_status and fashion_news_frequency are needed
# Aggregate transactions by customer_id
customer_purchase = transactions.groupby('customer_id').agg(total_purchases = ('article_id', 'count'),
                                                            total_spent = ('price', 'sum')).reset_index()
# Merge with customer information
customers_merged = customers.merge(customer_purchase, on = 'customer_id', how = 'left')
customers_merged.fillna({'total_purchases': 0, 'total_spent': 0}, inplace = True)

# Encode the values of each column into numerical values
customers_merged['club_member_status_num'] = customers_merged['club_member_status'].astype('category').cat.codes
customers_merged['fashion_news_frequency_num'] = customers_merged['fashion_news_frequency'].astype('category').cat.codes
#customers_merged

# Check the correlation
correlation_results = customers_merged[['club_member_status_num', 'fashion_news_frequency_num',
                                        'total_purchases', 'total_spent']].corr()
correlation_results

# Remove club_member_status and fashion_news_frequency as they are not good predictor
customers = customers.drop(columns = ['club_member_status', 'fashion_news_frequency', 'postal_code'])
customers

### Impute the NA age values in the customers dataset
## Identify customers who are not in the transaction history
whole_customers = set(customers['customer_id'])
tran_customers = set(transactions['customer_id'].unique())
no_tran_customers = whole_customers - tran_customers

## Identify customers with missing values of age
customers_na_age = set(customers[customers['age'].isna()]['customer_id'])

## Check if non-transactional customers are a subset of customers with NA age values
na_age_subset = no_tran_customers.issubset(customers_na_age)
difference_count = len(customers_na_age - no_tran_customers)
print(f"All non-transactional customers are in the missing age set: {na_age_subset}")
print(f"Number of customers with transaction history but missing age: {difference_count}")

import dask.dataframe as dd

## Because transaction dataset is too big (31M+ rows), decided to use dd instead of pandas
# Convert Pandas to Dask DataFrames

# transactions
transactions_dd = dd.read_csv('/content/transactions_train.csv', usecols=['t_dat','customer_id', 'article_id'])
def stratified_sample(group, frac=0.7, seed=42):
    np.random.seed(seed)
    return group.sample(frac=frac, random_state=seed)

transactions_dd = transactions_dd.groupby('customer_id').apply(stratified_sample, meta=transactions_dd)
transactions_dd = transactions_dd.reset_index(drop=True)

# customers
customers_dd = dd.read_csv('/content/customers.csv', usecols = ['customer_id', 'age'])
customers = customers_dd.compute()

# articles
articles_dd = dd.read_csv('/content/articles.csv', usecols = ['article_id', 'product_code'])
articles = articles_dd.compute()

# Perform Dask Merging
transactions_merged = transactions_dd.merge(customers_dd, on='customer_id', how='left')
transactions_merged = transactions_merged.merge(articles_dd, on='article_id', how='left')

# Convert back to Pandas only if needed
transactions_merged = transactions_merged.persist()
transactions_final = transactions_merged.compute()

# Check if num of rows is matched
len(transactions_final['customer_id'].unique())

# Step 1: Separate Customers with Known and Unknown Age
customers_with_known_age_dd = transactions_final_dd[~transactions_final_dd['age'].isna()]
customers_with_unknown_age_dd = transactions_final_dd[transactions_final_dd['age'].isna()]

# Step 2: Compute Average Age for Each Product (Dask Optimized)
average_age_per_product_dd = customers_with_known_age_dd.groupby('article_id')['age'].mean().compute()

# Step 3: Impute Age for Customers with Missing Values
customers_with_unknown_age_dd = customers_with_unknown_age_dd.map_partitions(
    lambda df: df.assign(imputed_age = df['article_id'].map(average_age_per_product_dd)
                         .fillna(customers['age'].median()))
)

# Step 4: Compute the Dask DataFrame to Get the Imputed Ages
customers_with_unknown_age_dd = customers_with_unknown_age_dd.compute()

# Step 5: Aggregate Imputed Ages per Customer
imputed_ages_dd = customers_with_unknown_age_dd.groupby('customer_id')['imputed_age'].mean()

# Step 6: Update the Original Customers Dataset with the Imputed Ages
customers_dd = customers_dd.map_partitions(
    lambda df: df.assign(age = df['customer_id'].map(imputed_ages_dd).fillna(df['age']))
)

# Step 7: Convert Back to Pandas If Needed
customers = customers_dd.compute()

# Step 8: Check for Remaining Missing Age Values
remaining_na_count = customers['age'].isna().sum()
print(f"Remaining missing age values: {remaining_na_count}")

# Compute the overall median age from customers with known age
overall_median_age = customers_dd['age'].dropna().median_approximate().compute()

# Fill remaining NA values in the 'age' column
customers_dd = customers_dd.map_partitions(lambda df: df.assign(age = df['age'].fillna(overall_median_age)))

# Convert back to pandas if needed
customers = customers_dd.compute()

# Check again for any remaining missing values in age
remaining_na_count = customers['age'].isna().sum()
print(f"Remaining missing age values: {remaining_na_count}") # Check num of remaining missing values is 0

# Recheck missing values after cleaning finished
missing_values_customers = customers.isnull().sum()
missing_values_articles = articles.isnull().sum()
missing_values_transactions = transactions.isnull().sum()

missing_values_summary = pd.DataFrame({
    'Dataset': (
        ['Customers'] * len(missing_values_customers) +
        ['Articles'] * len(missing_values_articles) +
        ['Transactions'] * len(missing_values_transactions)
    ),
    'Column': (
        list(missing_values_customers.index) +
        list(missing_values_articles.index) +
        list(missing_values_transactions.index)
    ),
    'Missing Values': (
        list(missing_values_customers.values) +
        list(missing_values_articles.values) +
        list(missing_values_transactions.values)
    ),
    'Missing Percentage': (
        list((missing_values_customers / len(customers) * 100).values) +
        list((missing_values_articles / len(articles) * 100).values) +
        list((missing_values_transactions / len(transactions) * 100).values)
    )
})
missing_values_summary

# Encode customer_id and article_id
customer_id_map = {cid: idx for idx, cid in enumerate(customers['customer_id'].unique())}
article_id_map = {aid: idx for idx, aid in enumerate(articles['article_id'].unique())}

transactions['customer_id'] = transactions['customer_id'].map(customer_id_map)
transactions['article_id'] = transactions['article_id'].map(article_id_map)

# Temporal Train-Test Split
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
split_date = transactions['t_dat'].max() - pd.Timedelta(days = 7)

train_data = transactions[transactions['t_dat'] < split_date]
test_data = transactions[transactions['t_dat'] >= split_date]

# Generate Labels
train_data['purchased'] = 1
test_data['purchased'] = 1

# Convert Pandas DataFrames to Dask DataFrames
transactions_dd = dd.from_pandas(transactions, npartitions=8)
train_data_dd = dd.from_pandas(train_data, npartitions=8)

# Get unique article IDs as a Dask array for efficient random sampling
all_article_ids = transactions_dd['article_id'].unique().compute()
all_article_ids = np.array(all_article_ids)

from scipy.sparse import csr_matrix

# Step 1: Precompute Customer-Item Interaction Matrix
customer_map = {cid: idx for idx, cid in enumerate(train_data['customer_id'].unique())}
article_map = {aid: idx for idx, aid in enumerate(train_data['article_id'].unique())}

train_data['customer_idx'] = train_data['customer_id'].map(customer_map)
train_data['article_idx'] = train_data['article_id'].map(article_map)

num_customers = len(customer_map)
num_articles = len(article_map)

# Create a sparse matrix for fast lookup
interaction_matrix = csr_matrix(
    (np.ones(len(train_data)), (train_data['customer_idx'], train_data['article_idx'])),
    shape=(num_customers, num_articles)
)
print(interaction_matrix)

# Step 2: Generate Negative Samples Efficiently
def fast_negative_sampling(customer_ids, num_negatives=5):
    np.random.seed(42)
    negative_samples = []

    for customer in customer_ids:
        purchased_articles = interaction_matrix[customer].indices
        negative_articles = np.setdiff1d(np.arange(num_articles), purchased_articles, assume_unique=True)

        # Fast negative sampling
        sampled_articles = np.random.choice(negative_articles, num_negatives, replace=False)

        negative_samples.extend([[customer, article, 0] for article in sampled_articles])

    return pd.DataFrame(negative_samples, columns=['customer_idx', 'article_idx', 'purchased'])

 # Step 3: Process in Parallel
customer_batches = np.array_split(np.arange(num_customers), 16)  # Increase batch size for faster processing
negative_samples_list = [fast_negative_sampling(batch) for batch in customer_batches]

# Merge Negative Samples
negative_samples = pd.concat(negative_samples_list)
