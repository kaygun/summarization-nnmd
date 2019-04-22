(defproject summary "0.1.0-SNAPSHOT"
  :description "Summarization"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [clatrix "0.5.0"]
                 [org.clojure/data.json "0.2.6"]
                 [org.apache.opennlp/opennlp-tools "1.9.1"]]
  :main summary 
  :aot [summary]
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
