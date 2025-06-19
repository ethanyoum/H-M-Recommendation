!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c h-and-m-personalized-fashion-recommendations
!unzip h-and-m-personalized-fashion-recommendations.zip -d /content/
!ls /content/

# articles file
text_path = '/content/articles.csv'
articles = pd.read_csv(text_path)
print(articles.shape)
articles.head(5)

# customers file
text_path2 = '/content/customers.csv'
customers = pd.read_csv(text_path2)
print(customers.shape)
customers.head(5)

# transactions file
text_path3 = '/content/transactions_train.csv'
transactions = pd.read_csv(text_path3)
transactions = transactions.groupby('customer_id').apply(lambda x: x.sample(frac=0.7, random_state=42)) # Reduce size of the data
transactions = transactions.reset_index(drop=True)
print(transactions.shape)
transactions.head(5)

# image file
image_dir = '/content/images'
files = os.listdir(image_dir)
print(files)

# Check some products in image
import glob
from PIL import Image
image_files = glob.glob(f'{image_dir}/**/*.*', recursive = True)
print(image_files)

for img_path in image_files[0:10]:
  image = Image.open(img_path)
  plt.imshow(image)
  plt.axis('off')
  plt.show()