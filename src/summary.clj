(ns summary
  (:import opennlp.tools.sentdetect.SentenceDetector
           opennlp.tools.sentdetect.SentenceDetectorME
           opennlp.tools.sentdetect.SentenceModel
           opennlp.tools.stemmer.PorterStemmer
           java.io.File)
  (:require [clojure.string :as st]
            [clojure.core.matrix :as cm]
            [clatrix.core :as cc]
            [clojure.java.io :as io]
            [clojure.data.json :as json])
  (:gen-class))

(defn bag-of-words [sentence stemmer stop-words]
   {sentence (as-> sentence $ 
                   (st/lower-case $) 
                   (st/replace $ #"[^\s\p{Isletter}]" "")
                   (st/split $ #"\s+") 
                   (filter #(not (stop-words %)) $) 
                   (map #(.stem stemmer %) $)
                   (into #{} $))})

(defn get-matrix [sentences detector stemmer stop-words]
   (let [raw (into {} (mapcat #(bag-of-words % stemmer stop-words) sentences))
         ws (->> (vals raw) (reduce concat) (into #{}) (into []))
         n (count sentences)
         m (count ws)
         A (cc/zeros m n)]
      (doseq [i (range n)]
         (doseq [w (get raw (nth sentences i))]
            (cc/set A (.indexOf ws w) i 1)))
      A))

(defn cost-fn [A B]
  (->> (cm/sub A B)
       (cm/to-vector)
       (map (fn [x] (* x x)))
       (reduce +)))

(defn random-matrix [n m]
  (as-> (repeatedly rand) $
    (take (* n m) $)
    (cm/reshape $ [n m])))

(defn nnmd [D k cost-fn epocs tol rate]
  (let [n (cm/row-count D)
        m (cm/column-count D)
        s (* n m)]
    (loop [W (random-matrix n k)
           H (random-matrix k m)
           i epocs
           c tol]
      (if (or (= i 0) (< c tol)) 
        [W H i c]
        (let [u (cm/reshape (take s (repeat 1)) [n m])
              Wt (cm/transpose W)
              Ht (cm/transpose H)
              et (cm/mul rate (cm/div W (cm/mmul u Ht)))
              mu (cm/mul rate (cm/div H (cm/mmul Wt u)))
              temp (cm/sub (cm/div D (cm/mmul W H)) u)]
          (recur (cm/add W (cm/mul et (cm/mmul temp Ht)))
                 (cm/add H (cm/mul mu (cm/mmul Wt temp)))
                 (dec i)
                 (/ (cost-fn D (cm/mmul W H)) s)))))))

(defn -main [text k-raw epochs-raw output]
   (let [epochs (read-string epochs-raw)
         k (read-string k-raw)
         sd (SentenceDetectorME. (SentenceModel. (File. "resources/en-sent.bin")))
         stemmer (PorterStemmer.)
         sentences (->> (slurp text) (.sentDetect sd) (into []))
         stop-words (as-> (slurp "resources/remove-en") $
                          (st/replace $ #"\p{IsPunctuation}" "")
                          (st/split $ #"\s+")
                          (into #{} $))
         matrix (get-matrix sentences sd stemmer stop-words)
         [W H i c] (cc/t (nnmd matrix k cost-fn epochs 1e-2 1e-2))
         res (map (fn [s w] {:sentence s :weight (into [] w)}) sentences (cc/matrix W))]
     (println W)
     (println H)
     (with-open [out (io/writer output)]
        (json/write res out))))

