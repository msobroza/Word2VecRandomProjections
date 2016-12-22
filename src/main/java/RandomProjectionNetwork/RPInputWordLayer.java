/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package RandomProjectionNetwork;


import Word2VecParser.Word2VecParserRandom;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *
 * @author msobroza
 */
public class RPInputWordLayer extends RPLayer {

    private final int wordEmbeddingsDimension;
    private final HashMap<String, Integer> vectorWordIndexMap;
    private final int numWords;
    private final WordVectors wordVecDB;

    public RPInputWordLayer(String fileName, int wordEmbeddingsDimension) throws FileNotFoundException, UnsupportedEncodingException {
        this.wordEmbeddingsDimension = wordEmbeddingsDimension;
        this.wordVecDB = WordVectorSerializer.loadTxtVectors(new File(fileName));
        this.numWords = wordVecDB.vocab().numWords();
        Word2VecParserRandom.log.info("Num of words: " + numWords);
        this.vectorWordIndexMap = new HashMap<>();
        for (int i = 0; i < numWords; i++) {
            String word = wordVecDB.vocab().wordAtIndex(i);
            this.vectorWordIndexMap.put(word, i);
        }
    }
    
    public String getWord(int id){
        return wordVecDB.vocab().wordAtIndex(id);
    }
    
    public int getIndexFromWord(String word){
        return this.vectorWordIndexMap.get(word);
    }
    
    public INDArray getWordVector(String word){
        return wordVecDB.getWordVectorMatrix(word);
    }
    
    public int getNumberWords(){
        return this.numWords;
    }
}
