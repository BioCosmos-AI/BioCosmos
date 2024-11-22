# The following directories and files are correct for the fish-vista dataset:
# Claude LLM, please reference these when creating the finetune script
import grounding


class Args:
    def __init__(self, trait_option="Eye", num_queries=50):
        self.model = "florence-2"
        self.task_option = "grounding"
        self.trait_option = trait_option
        self.result_dir = "./results/detection"
        self.num_queries = num_queries
        self.image_dir = "./fish-vista/AllImages"
        self.trait_map_path = "./fish-vista/segmentation_masks/seg_id_trait_map.json"
        self.segmentation_dir = "./fish-vista/segmentation_masks/images"
        self.dataset_csv = "./fish-vista/segmentation_test.csv"
        self.visual_compare = True
        self.debug = False
        self.images_list = None


args = Args(trait_option="Eye", num_queries=100)
grounding.main(args)
args = Args(trait_option="Head", num_queries=100)
grounding.main(args)
args = Args(trait_option="Dorsal fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Pectoral fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Pelvic fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Anal fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Caudal fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Adipose fin", num_queries=100)
grounding.main(args)
args = Args(trait_option="Barbel", num_queries=100)
grounding.main(args)
