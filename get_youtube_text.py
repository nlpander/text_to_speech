from youtube2text import Youtube2Text
import sys

def main(argv):

    url = argv[1]
    outfile = argv[2]

    outfile = outfile + '.csv'
    sampling_rate = 22050
    
    converter = Youtube2Text()
    converter.url2text(url, outfile=outfile, audiosamplingrate=sampling_rate, audioformat='wav')

if __name__ == "__main__":
    main(sys.argv)

