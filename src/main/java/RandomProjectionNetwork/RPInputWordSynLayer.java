/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;

import Tools.SynonymDict;
import static Word2VecParser.Word2VecParserRandom.INPUT_SYNONYM;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.HashSet;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class RPInputWordSynLayer extends RPInputWordLayer {

    private final int minimumSamplesPerPattern;
    private final SynonymDict syn;

    public RPInputWordSynLayer(String fileNameEmbedding, String fileNameSyn, int wordEmbeddingsDimension, int minimumSamplesPerPattern) throws FileNotFoundException, UnsupportedEncodingException {
        super(fileNameEmbedding, wordEmbeddingsDimension);
        this.syn = new SynonymDict(fileNameSyn);
        this.minimumSamplesPerPattern = minimumSamplesPerPattern;
    }

    public HashMap<String, INDArray> getVectorSyns(String word) {
        HashMap<String, INDArray> result = new HashMap<>();
        // Verify the minimum number of samples
        HashSet<String> wordSynonyms = syn.getSynonyms(word);
        if (wordSynonyms == null) {
            return null;
        }
        // Exclude the word it self from results
        wordSynonyms.remove(word);
        // Exclude words that it does not exists a representation in word embeddings
        HashSet<String> wordSynonymsCopy;
        wordSynonymsCopy = (HashSet<String>) wordSynonyms.clone();
        for (String s : wordSynonymsCopy) {
            if (!this.vectorWordIndexMap.containsKey(s)) {
                wordSynonyms.remove(s);
            }
        }
        if (wordSynonyms.size() >= minimumSamplesPerPattern) {
            for (String s : wordSynonyms) {
                result.put(s, this.getWordVector(word));
            }
            return result;
        } else {
            return null;
        }
    }

}
