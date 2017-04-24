import java.util.*;
import java.io.*;

public class ParseLabels {
    public static void main(String[] args) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(new File("train_labels.csv"));
        Scanner scan = new Scanner(new File("predata.csv"));
        scan.nextLine();
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
