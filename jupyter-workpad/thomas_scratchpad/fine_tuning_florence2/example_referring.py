import referring


class Args:
    def __init__(self, trait_option="Eye", num_queries=50):
        self.model = "florence-2"
        self.task_option = "referring"
        self.trait_option = trait_option
        self.result_dir = "./results/referring"
        self.num_queries = num_queries
        self.image_dir = "./fish-vista/AllImages"
        self.trait_map_path = "./fish-vista/segmentation_masks/seg_id_trait_map.json"
        self.segmentation_dir = "./fish-vista/segmentation_masks/images"
        self.dataset_csv = "./fish-vista/segmentation_test.csv"
        self.visual_output = True
        self.debug = False


args = Args(trait_option="Eye", num_queries=100)
referring.main(args)
args = Args(trait_option="Head", num_queries=100)
referring.main(args)
args = Args(trait_option="Dorsal fin")
referring.main(args)
args = Args(trait_option="Pectoral fin")
referring.main(args)
args = Args(trait_option="Pelvic fin")
referring.main(args)
args = Args(trait_option="Anal fin")
referring.main(args)
args = Args(trait_option="Caudal fin")
referring.main(args)
args = Args(trait_option="Adipose fin")
referring.main(args)
args = Args(trait_option="Barbel")
referring.main(args)
