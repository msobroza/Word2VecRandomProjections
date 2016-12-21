/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Model.Embedding;

import java.util.Objects;

/**
 *
 * @author msobroza
 */
public abstract class Embedding {

    protected String word;
    
    
    public Embedding(){
    }
    
    public Embedding(String word) {
        this.word = word;
    }

    public abstract Object getVector();

    public String getWord() {
        return this.word;
    }

    @Override
    public boolean equals(Object o) {
        return ((Embedding) o).word.equals(this.word);
    }

    @Override
    public int hashCode() {
        int hash = 5;
        hash = 37 * hash + Objects.hashCode(this.word);
        return hash;
    }

}
