# Document Summarization via Nonnegative Matrix Factorization

## Description of the problem

Assume we have a matrix $A$ of size $n\times m$ which consists of nonnegative entries.  We want to write $A$
as a product $A = BC$ where $B$ has size $n\times k$ and $C$ has size $k\times m$ where $k$ is much smaller
than both $n$ and $m$.  We also want $B$ and $C$ to consist nonnegative entries.

We can think of this problem as an optimization problem where the error function is 
$$ err(B,C) = \sum_{i,j,\ell} |a_{ij}-b_{i\ell}c_{\ell j}| $$
Today, I am going to implement a solution using clojure.  I am going to apply this to a problem coming from
natural language processing.

## Nonnegative Matrix Decomposition and Document Summarization

Assume we have a text, and we write a matrix $A$ whose rows are labeled with the sentences appearing in the
text and whose columns are labeled with the words appearing in the text.  For a (sentence,word)-pair the
corresponding entry in the matrix is 1 if the word appears in the sentence, and the entry is 0 otherwise.
Dividing each row by the sum of the terms in that row, we convert these 1's and 0's to a probability distribution:
in the new matrix for a (sentence,word)-pair the corresponding entry is the probability that that word
appears in that sentence.

If we apply the nonnegative matrix procedure to this matrix we get two matrices $B$ and $C$ such that 
$A = BC$.  I will use the following hypothesis: a *topic* is a specific probability distribution over 
the set of all words appearing in a text.  Thus such a decomposition tries to identify k-many topics
that one can associate with the text at hand.  While $B$ measures how much of each sentence belongs to
a topic, $C$ measures the same thing for each word.

## An Implementation

I am going to re-cycle the code I wrote in my earlier posts: one for [Latent Semantic Analysis][1]
to get the relevant matrix from a document, and another for [Nonnegative Matrix Decomposition][2]
in clojure:

[1]: https://kaygun.tumblr.com/post/184320283959/latent-semantic-analysis-in-clojure
[2]: https://kaygun.tumblr.com/post/179635625399/nonnegative-matrix-decomposition-in-clojure

First, let us define our namespace with the necessary libraries:
```clojure
(ns summary
  (:import opennlp.tools.sentdetect.SentenceDetector
           opennlp.tools.sentdetect.SentenceDetectorME
           opennlp.tools.sentdetect.SentenceModel
           opennlp.tools.stemmer.PorterStemmer
           java.io.File)
  (:require [clojure.string :as st]
            [clojure.core.matrix :as cm]
            [clatrix.core :as cc])
  (:gen-class))
```
```clojure

```
Now, let us write the functions that create the matrix $A$ from a given document:
```clojure
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
```
```clojure
#'summary/bag-of-words
#'summary/get-matrix
```
For the matrix decomposition, first I need the error function and a random matrix function:
```clojure
(defn cost-fn [A B]
  (->> (cm/sub A B)
       (cm/to-vector)
       (map (fn [x] (* x x)))
       (reduce +)))

(defn random-matrix [n m]
  (as-> (repeatedly rand) $
    (take (* n m) $)
    (cm/reshape $ [n m])))
```
```clojure
#'summary/cost-fn
#'summary/random-matrix
```
Now, the matrix decomposition code:
```clojure
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
```
```clojure
#'summary/nnmd
```
So, let us test:
```clojure
(def summary
   (let [sd (SentenceDetectorME. (SentenceModel. (File. "resources/en-sent.bin")))
         stemmer (PorterStemmer.)
         sentences (->> (slurp "data/textc") (.sentDetect sd) (into []))
         stop-words (as-> (slurp "resources/remove-en") $
                          (st/replace $ #"\p{IsPunctuation}" "")
                          (st/split $ #"\s+")
                          (into #{} $))
         matrix (get-matrix sentences sd stemmer stop-words)
         [W H i c] (cc/t (nnmd matrix 3 cost-fn 2000 1e-2 1e-2))
         weights (cc/matrix W)]
       (map (fn [s w] {:sentence s :weight w}) sentences weights)))
```



|Topic 1|Topic 2|Topic 3|Sentence|
|:---|:---|:---|:-------------------------|
|0.00|0.18|0.36|	The Obama administration has backed down in its bitter dispute with Silicon Valley over the encryption of data on iPhones and other digital devices, concluding that it is not possible to give American law enforcement and intelligence agencies access to that information without also creating an opening that China, Russia, cybercriminals and terrorists could exploit.|
|0.00|0.00|0.18|	With its decision, which angered the FBI and other law enforcement agencies, the administration essentially agreed with Apple, Google, Microsoft and a group of the nation’s top cryptographers and computer scientists that millions of Americans would be vulnerable to hacking if technology firms and smartphone manufacturers were required to provide the government with “back doors,” or access to their source code and encryption keys.|
|0.00|0.18|0.00|	Companies like Apple say they are protecting their customers’ information by resisting government demands for access to text messages.|
|0.00|0.18|0.00|	A standoff has grown between the sides as the companies have embraced tougher encryption.|
|0.00|0.18|0.00|	Peter G Neumann, a computer security pioneer, says “there are more vulnerabilities than ever.|
|0.00|0.00|0.18|	Security experts like Richard A. Clarke, the former White House counterterrorism czar, also signed the letter to Obama.|
|0.00|0.00|0.18|	That would enable the government to see messages, photographs and other data now routinely encrypted on smartphones.|
|0.00|0.00|0.18|	Current technology puts the keys for access to the information in the hands of the individual user, not the companies.|
|0.00|0.75|0.87|	The first indication of the retreat came on Thursday, when the FBI director, James B Comey, told the Senate Homeland Security and Governmental Affairs Committee that the administration would not seek legislation to compel the companies to create such a portal.|
|0.00|0.18|0.18|	But the decision, made at the White House a week ago, goes considerably beyond that.|
|0.19|0.00|0.00|	While the administration said it would continue to try to persuade companies like Apple and Google to assist in criminal and national security investigations, it determined that the government should not force them to breach the security of their products.|
|0.79|0.93|0.65|	In essence, investigators will have to hope they find other ways to get what they need, from data stored in the cloud in unencrypted form or transmitted over phone lines, which are covered by a law that affects telecommunications providers but not the technology giants.|
|0.00|0.18|0.00|	Mr Comey had expressed alarm a year ago after Apple introduced an operating system that encrypted virtually everything contained in an iPhone.|
|0.00|0.00|0.18|	What frustrated him was that Apple had designed the system to ensure that the company never held on to the keys, putting them entirely in the hands of users through the codes or fingerprints they use to get into their phones.|
|0.38|0.00|0.00|	As a result, if Apple is handed a court order for data — until recently, it received hundreds every year — it could not open the coded information.|
|0.19|0.00|0.18|	Mr Comey compared that system to the creation of a door no law officers could enter, or a car trunk they could not unlock.|
|0.19|0.00|0.00|	His concern about what the FBI calls the “going dark” problem received support from the director of the National Security Agency and other intelligence officials.|
|0.19|0.00|0.00|	But after a year of study and extensive White House debate, President Obama and his advisers have reached a broad conclusion that an effort to compel the companies to give the government access would fail, both politically and technologically.|
|0.95|0.00|0.73|	“This looks promising, but there’s still going to be tremendous pressure from law enforcement,” said Peter G Neumann, one of the nation’s leading computer scientists and a co-author of a paper that examined the government’s proposal for special access.|
|0.19|0.18|0.00|	“The N.S.A. is capable of dealing with the cryptography for now, but law enforcement is going to have real difficulty with this.|
|0.19|0.00|0.00|	This is never a done deal.”|
|0.19|0.00|0.00|	In the paper, released in July, Mr Neumann and other top cryptographers and computer scientists argued that there was no way for the government to have a back door into encrypted communications without creating an opening that would be exploited by Chinese and Russian intelligence agents, cybercriminals and terrorist groups.|
|0.00|0.00|0.18|	Inside the White House, the Office of Science and Technology Policy came largely to the same conclusion.|
|0.19|0.00|0.00|	Those determinations surprised the FBI and local law enforcement officials, who had believed just months ago that the White House would ultimately embrace their efforts.|
|0.00|0.16|1.29|	The intelligence agencies were less vocal, which may reflect their greater capability to search for and gather information.|
|0.00|0.18|0.00|	The National Security Agency spends vast sums to get around digital encryption, and it has tools and resources that local law enforcement officials still do not have and most likely never will.|
|0.38|0.00|0.00|	Disclosures by the former N.S.A. contractor Edward J. Snowden showed the extent of the agency’s focus on cracking and circumventing the encryption of digital communications, including those of Apple, Facebook, Google and Yahoo users.|
|0.00|0.00|0.18|	There were other motivations for the administration’s decision.|
|0.19|0.00|0.00|	Mr Obama and his aides had come to fear that the United States could set a precedent that China and other nations would emulate, requiring Apple, Google and the rest of America’s technology giants to provide them with the same access, officials said.|
|0.38|0.00|0.00|	Timothy D Cook, the chief executive of Apple, sat at the head table with Mr Obama and Xi Jinping, the Chinese president, at a state dinner at the White House last month.|
|0.19|0.00|0.00|	According to government officials and industry executives, Mr Cook told Mr Obama that the Chinese were waiting for an opportunity to seize on administration action to insist that Apple devices, which are also encrypted in China, be open to Beijing’s agents.|
|0.00|0.00|0.18|	In January, three months after Mr Comey began pressing companies for special government access, Chinese officials had threatened to do just that: They considered submitting foreign companies to invasive audits and requiring them to build back doors into their hardware and software.|
|0.00|0.18|0.00|	Those rules have not been put into effect.|
|0.00|0.18|0.00|	The Obama administration’s position was also undercut by officials’ inability to keep their own data safe from Chinese hackers, as shown by the extensive cyberattack at the Office of Personnel Management discovered this year.|
|0.00|0.18|0.18|	That breach, and its aftermath, called into question whether the government could keep the keys to the world’s communications safe from its adversaries in cyberspace.|
|0.15|0.46|0.29|	White House officials said they would continue trying to persuade technology companies to help them in investigations, but they did not specify how.|
|0.00|0.18|0.00|	“As the president has said, the United States will work to ensure that malicious actors can be held to account, without weakening our commitment to strong encryption,” said Mark Stroh, a spokesman for the National Security Council.|
|0.38|0.00|0.00|	“As part of those efforts, we are actively engaged with private companies to ensure they understand the public safety and national security risks that result from malicious actors’ use of their encrypted products and services.|
|0.00|0.00|0.18|	However, the administration is not seeking legislation at this time.”|
|0.00|1.05|0.74|	But here in Silicon Valley, executives did not think the government’s announcement went far enough.|
|0.00|0.00|0.18|	According to administration officials and technology executives, Mr Cook of Apple has pressed the White House for a clear statement that it will never seek a back door in any form, legislative or technical — a statement he hoped to take to Beijing, Moscow and even London.|
|0.00|0.18|0.00|	Prime Minister David Cameron of Britain has threatened to ban encrypted devices and services, like the iPhone and Facebook’s popular WhatsApp messaging service, but has done nothing so far to make good on that threat.|
|0.19|0.00|0.00|	Technology executives are determined to reassure customers abroad that American intelligence agencies are not reading their digital communications.|
|0.00|0.14|0.22|	It is an effort driven by economics: 64 percent of Apple’s revenue originates overseas.|
|0.00|0.00|0.18|	Apple, Google, Facebook and Microsoft argue that people put not only their conversations but their entire digital lives — medical records, tax returns, bank accounts — into a device that slips into their pocket.|
|0.00|0.00|0.37|	While Mr Obama has repeatedly said he is sympathetic to the concerns of law enforcement officials, he made clear during a visit to Silicon Valley in February that he was also aware of privacy concerns and that he sought to balance both interests.|
|0.00|0.00|0.18|	Technologists responded that, with regard to encryption, no such balance existed.|
|0.00|0.00|0.18|	“The real problem is, I don’t see any middle ground for dumbing down everything to make special access possible and having the secure systems we need for commerce, government and everything else,” Mr Neumann said.|

In this example, the third topic seems promising.  If we take only the sentences of weight 0.3 and higher we get

> The Obama administration has backed  down in its bitter dispute with
> Silicon  Valley over  the encryption  of data  on iPhones  and other
> digital devices, concluding that it is not possible to give American
> law enforcement and intelligence agencies access to that information
> without also creating an  opening that China, Russia, cybercriminals
> and terrorists could  exploit.  The first indication  of the retreat
> came on  Thursday, when the  FBI director,  James B Comey,  told the
> Senate Homeland Security and Governmental Affairs Committee that the
> administration would not seek legislation to compel the companies to
> create such a  portal.  In essence, investigators will  have to hope
> they find other ways to get what  they need, from data stored in the
> cloud in unencrypted form or transmitted over phone lines, which are
> covered by a  law that affects telecommunications  providers but not
> the  technology giants.   “This looks  promising, but  there’s still
> going to be tremendous pressure  from law enforcement,” said Peter G
> Neumann,  one of  the  nation’s leading  computer  scientists and  a
> co-author of  a paper  that examined  the government’s  proposal for
> special access.   The intelligence  agencies were less  vocal, which
> may  reflect  their greater  capability  to  search for  and  gather
> information.  But here  in Silicon Valley, executives  did not think
> the government’s announcement  went far enough.  While  Mr Obama has
> repeatedly said he is sympathetic to the concerns of law enforcement
> officials,  he  made clear  during  a  visit  to Silicon  Valley  in
> February that  he was  also aware  of privacy  concerns and  that he
> sought to balance both interests.

