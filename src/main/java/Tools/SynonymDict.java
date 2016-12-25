/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Tools;

import Word2VecParser.Config;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

/**
 *
 * @author msobroza
 */
public class SynonymDict {

    private final ArrayList<String> wordsList;
    private final HashMap<Integer, HashSet<String>> synonymsMap;

    public SynonymDict() {
        this.wordsList = new ArrayList<>();
        this.synonymsMap = new HashMap<>();
    }

    public SynonymDict(String fileName) {
        this.wordsList = new ArrayList<>();
        this.synonymsMap = new HashMap<>();
        this.addSynonymsFromFile(fileName);
    }

    public HashSet<String> getSynonyms(String word) {
        int id = this.wordsList.indexOf(word);
        if (id != -1) {
            return getSynonyms(id);
        }else{
            return null;
        }
    }

    public HashSet<String> getSynonyms(int idWord) {
        return this.synonymsMap.get(idWord);
    }

    public void addSynonymsFromFile(String fileName) {
        if (FileIO.fileExists(fileName)) {
            List<String[]> lines = FileIO.readFile(fileName);
            for (String[] l : lines) {
                int idWord = Integer.parseInt(l[Config.SynonymFile.IndexWord.getIndex()]);
                String word = l[Config.SynonymFile.Word.getIndex()];
                for (int i = Config.SynonymFile.FistSynonym.getIndex(); i < l.length; i++) {
                    addSynonym(word, idWord, l[i]);
                }
            }

        }
    }

    public void addSynonym(String word, int id, String syn) {
        HashSet<String> aux;
        int idWord = this.wordsList.indexOf(word);
        if (idWord == -1) {
            wordsList.add(id, word);
            idWord = wordsList.size() - 1;
            aux = new HashSet<>();
        } else {
            if (!this.synonymsMap.containsKey(idWord)) {
                aux = new HashSet<>();
            } else {
                aux = this.synonymsMap.get(idWord);
            }
        }
        aux.add(syn);
        this.synonymsMap.put(idWord, aux);
    }

    public void addSynonym(String word, String syn) {
        HashSet<String> aux;
        int idWord = this.wordsList.indexOf(word);
        if (idWord == -1) {
            wordsList.add(word);
            idWord = wordsList.size() - 1;
            aux = new HashSet<>();
        } else {
            if (!this.synonymsMap.containsKey(idWord)) {
                aux = new HashSet<>();
            } else {
                aux = this.synonymsMap.get(idWord);
            }
        }
        aux.add(syn);
        this.synonymsMap.put(idWord, aux);
    }

}
