from litepali import LitePali, ImageFile
from dotenv import load_dotenv
import os, time

litepali = LitePali()
load_dotenv()
index_path = os.getenv('INDEX_PATH')

image_paths = sorted([os.path.abspath(os.path.join(os.getenv('UI_PATH'), fname)) for fname in os.listdir(os.getenv('UI_PATH'))])

for idx, path in enumerate(image_paths):
    img_file = ImageFile(
        path=path,
        document_id=1,
        page_id=idx,
        file_validation=True
    )
    litepali.add(img_file)

print("\n### ColPali processing started ###")
start = time.time()
litepali.process()
print(f"\nColpali processing finished in {time.time()-start}")

results = litepali.search("Who is Illia?")
print("\n### Results of query: \n", results)

litepali.save_index(path=index_path, compressed=True)


