import java.util.*;
import java.io.*;

public class ParseLabels {
    public static void main(String[] args) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(new File("../../data/preprocess_data/train_labels.csv"));
        Scanner scan = new Scanner(new File("../../data/preprocess_data/predata.csv"));
        scan.nextLine();
        pw.println("filename,label");
        while (scan.hasNextLine()) {
            String[] vals = scan.nextLine().split(",");
            if (vals.length <= 1) {
                continue;
            }
            if (vals[1].equalsIgnoreCase("Y") || vals[1].equalsIgnoreCase("N")) {
                pw.println(vals[0] + "," + (vals[1].equalsIgnoreCase("Y") ? 1 : 0));
            }
        }
        pw.close();
    }
}
