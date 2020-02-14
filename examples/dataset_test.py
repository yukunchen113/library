#tests the dataset function
import utils
import utils.general_constants as gc

def main():
	print("celeba")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=False)
	print("celeba-hq 64")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=64)
	print("celeba-hq 128")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=128)
	print("celeba-hq 256")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=256)
	print("celeba-hq 512")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=512)
	print("celeba-hq 1024")
	dataset, get_group = utils.dataset.get_celeba_data(gc.datapath, group_num=1, save_new=False, get_group=True, shuffle=False, max_len_only=True, is_HD=1024)


	images_1, labels_1 = get_group()
	images_2, labels_2 = get_group()

if __name__ == '__main__':
	main()