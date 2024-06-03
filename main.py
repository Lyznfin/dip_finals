import argparse
from main_v import classify_v
from main_c import classify_c

def main():
    parser = argparse.ArgumentParser(description="Classify eye as either normal or cataract.")
    parser.add_argument("input_path", type=str, help="Path to the input image.")
    parser.add_argument("flavor", type=str, help="vanilla or choco (development)")
    args = parser.parse_args()
    match args.flavor:
        case 'vanilla':
            classify_v(args.input_path)
        case 'choco':
            classify_c(args.input_path)
        case _:
            return f"Wrong command"
    # print(args.input_path)

if __name__ == "__main__":
    main()
    # example usage
    # py main.py C:\Users\manzi\VSCoding\cataract_classification\image_test\image_cataract_1.png choco
    # py main.py C:\Users\manzi\VSCoding\cataract_classification\image_test\image_cataract_2.png vanilla
    # py main.py C:\Users\manzi\VSCoding\cataract_classification\image_test\normal_img_3.png choco