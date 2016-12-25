package Tools;

import Word2VecParser.Word2VecParserRandom;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class FileIO {

    private static String DELIMITER = " ";

    public static boolean fileExists(String filePath) {
        File file = new File(filePath);
        return file.exists();
    }
    
    public static List<String[]> readFile(String filePath) {
        List<String[]> result = new ArrayList<>();
        FileInputStream inputStream = null;
        Scanner sc = null;
        String sentence;
        String[] splitSentence;
        int count = 0;
        try {
            inputStream = new FileInputStream(filePath);
            sc = new Scanner(inputStream, "UTF-8");
            while (sc.hasNextLine()) {
                count += 1;
                if(count % 100000 == 0)
                    Word2VecParserRandom.log.debug("100000 more where read");
                sentence = sc.nextLine();
                splitSentence = sentence.split(DELIMITER);
                result.add(splitSentence);
            }
            if (sc.ioException() != null) {
                throw sc.ioException();
            }
        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException ex) {
                }
            }
            if (sc != null) {
                sc.close();
            }
        }
        return result;
    }

   
    public static HashMap<Integer, List<String>> readSplittedFile(String filePath) {
        HashMap<Integer, List<String>> result = new HashMap<>();
        FileInputStream inputStream = null;
        Scanner sc = null;
        String sentence;
        String[] splitSentence;
        List<String> listColumnAux;
        int count = 0;
        try {
            inputStream = new FileInputStream(filePath);
            sc = new Scanner(inputStream, "UTF-8");
            while (sc.hasNextLine()) {
                count += 1;
                if(count % 100000 == 0)
                    Word2VecParserRandom.log.debug("100000 more where read");
                sentence = sc.nextLine();
                splitSentence = sentence.split(DELIMITER);
                for (int i = 0; i < splitSentence.length; i++) {
                    if (result.containsKey(i)) {
                        listColumnAux = (LinkedList<String>) result.get(i);
                    } else {
                        listColumnAux = new LinkedList<>();
                        result.put(i, listColumnAux);
                    }
                    listColumnAux.add(splitSentence[i]);
                }
            }
            if (sc.ioException() != null) {
                throw sc.ioException();
            }
        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        } finally {
            if (inputStream != null) {
                try {
                    inputStream.close();
                } catch (IOException ex) {
                }
            }
            if (sc != null) {
                sc.close();
            }
        }
        return result;
    }

    public static boolean createCsvFile(String filePath, String[] columns) {

        try {
            String line = "";
            BufferedWriter StrW = new BufferedWriter(new FileWriter(filePath, true));
            for (int i = 0; i < columns.length; i++) {
                line = line + columns[i] + ";";
            }
            StrW.write(line + "\n");
            StrW.flush();
            StrW.close();
            return true;
        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return false;
    }
    
     public static void bufferedWrite(List<String> content, String filePath) {

        Path fileP = Paths.get(filePath);
        Charset charset = Charset.forName("utf-8");

        try (BufferedWriter writer = Files.newBufferedWriter(fileP, charset)) {

            for (String line : content) {
                writer.write(line, 0, line.length());
                writer.newLine();
            }

        } catch (IOException e) {
        }
    }
    
    
    
}
