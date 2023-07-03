# Test Thema
von *Max Mustermann, Berta Beispiel und Peter Lustig*

Abstract beschreibt kurz und in wenigen Sätzen ihr Thema. Der Abstract dient dem Lesenden als Orientierungshilfe ob er/sie weiterlesen möchten.

Die folgenden Gliederung ist Beispielhaft und kann von Ihnen nach Wunsch angepasst werden. Am Ende sollte ein ca. 10-15 Seiten langes Dokument vorliegen. Falls Sie weitere Infos zur Formatierung benötigen schauen Sie in der [Referenz](https://squidfunk.github.io/mkdocs-material/reference/).
## Einleitung / Motivation

## Stand der Forschung

## Methoden
Die Methoden zur Entwicklung von Text-to-Speech (TTS)-Systemen haben in den letzten Jahren erhebliche Fortschritte gemacht, indem sie fortschrittliche mathematische Modelle und Techniken einsetzen. In diesem Abschnitt werden wir auf einige der wichtigsten Methoden eingehen, die bei der Realisierung von TTS-Systemen verwendet werden.
### Hidden Markov Models (HMMs)
Hidden Markov Models (HMMs) sind eine grundlegende Methode in der Sprachverarbeitung und haben auch in der Entwicklung von TTS-Systemen eine bedeutende Rolle gespielt. HMMs bieten einen mathematischen Rahmen, um die statistischen Eigenschaften von Sprache zu modellieren und die Beziehung zwischen Text und Sprachsignalen zu erfassen.
Das zugrunde liegende mathematische Modell eines HMMs besteht aus einer Menge von Zuständen, Übergängen zwischen den Zuständen und Emissionen, die mit den Zuständen verknüpft sind. Für die Anwendung von HMMs im Bereich der TTS-Synthese werden typischerweise drei Arten von Zuständen definiert. Die Zustände des Emissionsmodells repräsentieren die Klänge oder Phoneme, die in der Sprache vorhanden sind. Jeder Zustand ist mit einer Wahrscheinlichkeitsverteilung über die möglichen akustischen Merkmale verbunden, die für das jeweilige Phonem charakteristisch sind. Die Zustände des Übergangsmodells repräsentieren die linguistische Struktur des Textes. Sie können Worte, Silben oder andere linguistische Einheiten sein. Die Übergänge zwischen den Zuständen des Übergangsmodells modellieren die statistische Wahrscheinlichkeit, mit der eine bestimmte linguistische Einheit auf eine andere folgt. Der Anfangszustand repräsentiert den Beginn des Textes oder der Sprachsequenz. Er gibt an, welche linguistische Einheit zuerst erzeugt wird.<br><br>
Die grundlegende Idee hinter HMMs besteht darin, dass der Übergang von einem Zustand zum nächsten stochastisch erfolgt, basierend auf den Übergangswahrscheinlichkeiten zwischen den Zuständen. Zusätzlich zu den Zustandsübergängen emittiert jeder Zustand eine bestimmte Wahrscheinlichkeitsverteilung über die akustischen Merkmale.
Bei der TTS-Synthese wird das HMM-Modell verwendet, um akustische Modelle zu erzeugen, die die Beziehung zwischen Text und Sprachsignalen erfassen. Der Text wird in eine Sequenz von Zuständen des Emissionsmodells übersetzt, und die HMM-Übergangswahrscheinlichkeiten werden verwendet, um die Reihenfolge und Dauer der Zustände zu bestimmen. Anhand der Wahrscheinlichkeitsverteilungen der akustischen Merkmale können dann Sprachsignale erzeugt werden, die dem Text entsprechen.<br><br>
Die Parameter des HMM-Modells, wie die Übergangswahrscheinlichkeiten und die Wahrscheinlichkeitsverteilungen der akustischen Merkmale, werden typischerweise mit Hilfe von Trainingsdaten geschätzt. Durch das Lernen aus großen Sprach Datensätzen kann das HMM-Modell verfeinert werden, um eine bessere Modellierung der Sprache zu erreichen und hochwertige Sprachsynthese zu ermöglichen.<br><br>
Obwohl HMMs eine bewährte Methode in der Sprachverarbeitung sind, haben sie auch ihre Einschränkungen. Insbesondere können sie Schwierigkeiten haben, komplexe linguistische Phänomene und Variabilität in den Sprachsignalen genau zu modellieren. Dennoch haben HMMs als grundlegende Methode in der TTS-Synthese einen wichtigen Beitrag geleistet und sind auch weiterhin Gegenstand der Forschung und Entwicklung, um ihre Leistung und Anwendungsbereiche zu verbessern.
### Deep Learning und NNs
Deep Learning und neuronale Netzwerke haben zu einer sprunghaften Verbesserung von TTS-Systemen geführt. Diese Methoden verwenden mehrschichtige neuronale Netzwerke, um komplexe Funktionen zu erlernen und hochdimensionale Daten zu verarbeiten. Im Bereich der TTS-Synthese können diese verwendet werden, um direkt Text zu Sprachsignalen abbilden zu können, ohne den Umweg über diskrete Zustände wie bei HMMs.<br><br>
Ein beliebter Ansatz ist die Verwendung von rekurrenten neuronalen Netzwerken (RNNs) oder deren Weiterentwicklungen wie den Long Short-Term Memory (LSTM) oder den Gated Recurrent Unit (GRU) Modellen. Diese Modelle haben die Fähigkeit, Sequenzdaten effektiv zu modellieren und können auf den TTS-Kontext angepasst werden, um Text in akustische Merkmale zu übersetzen.<br><br>
Ein weiterer wichtiger Fortschritt im Bereich des Deep Learning für TTS ist die Verwendung von Convolutional Neural Networks (CNNs). CNNs können verwendet werden, um akustische Merkmale auf verschiedenen Zeitskalen zu extrahieren und die Sprachsynthesequalität zu verbessern. Durch die Kombination von CNNs und RNNs können komplexe Zusammenhänge zwischen Text und Sprachsignalen erfasst werden.
### WaveNet und SampleRNN
WaveNet, entwickelt von DeepMind, nutzt atrous convolutions und autoregressive Architekturen, um hochauflösende Sprachsignale zu generieren. Atrous convolutions ermöglichen es dem Modell, Informationen auf verschiedenen Skalen zu erfassen, indem sie Lücken zwischen den Punkten des Filters einführen. Diese Technik ermöglicht es WaveNet, sowohl lokale als auch globale Abhängigkeiten in den Sprachsignalen zu modellieren, was zu realistischeren und qualitativ hochwertigen Audioausgaben führt. Allerdings erfordert die komplexe Struktur von WaveNet beträchtliche Rechenressourcen und das Training gestaltet sich aufgrund ihrer Komplexität als herausfordernd.<br><br>
Diese Modelle haben dazu beigetragen, die Qualität der Sprachsynthese erheblich zu verbessern und natürliche Sprachausgaben zu erzeugen. Sie erfordern jedoch erhebliche Rechenressourcen und können aufgrund ihrer komplexen Struktur nur schwer trainiert werden.
### Tacotron und Transformer-basierte Modelle
Tacotron ist ein Aufmerksamkeitsmechanismus-basiertes TTS-Modell, das auf RNNs oder Transformer-Architekturen basiert. Es ermöglicht die direkte Vorhersage von akustischen Merkmalen aus Text durch die Verwendung eines Aufmerksamkeits Mechanismus, der es dem Modell ermöglicht, sich auf die relevanten Teile des Textes zu konzentrieren. Transformer-basierte Modelle, wie beispielsweise das Transformer-TTS-Modell, haben ebenfalls beeindruckende Ergebnisse erzielt, indem sie auf der Transformer-Architektur aufbauen, die auf sequentielle Daten angewendet wird.<br><br>
Diese Modelle bieten eine gute Balance zwischen Sprachqualität und Effizienz. Sie können große Datensätze verwenden, um qualitativ hochwertige Sprachsynthese zu erzeugen und sind in der Lage, komplexe sprachliche Strukturen zu erfassen.
### Transfer Learning
Transfer Learning ist eine Methode, bei der vor trainierte Modelle auf einem großen allgemeinen Sprach-Datensatz verwendet und anschließend auf spezifische TTS-Aufgaben fein abgestimmt werden. Durch die Übertragung des gelernten Wissens aus den vor trainierten Modellen können TTS-Systeme von den Vorteilen der bereits erlernten Sprache Repräsentationen und -funktionen profitieren.<br><br>
Diese Methode ermöglicht eine schnellere Entwicklung von TTS-Systemen und erzielt oft gute Leistung, selbst mit begrenzten Trainingsdaten. Transfer Learning ist besonders nützlich, wenn spezifische Domänen oder Sprecher berücksichtigt werden sollen.<br><br>
Zusammenfassend haben wir einen umfassenden Überblick über verschiedene Methoden in der TTS-Forschung gegeben, darunter Hidden Markov Models (HMMs), Deep Learning und neuronale Netzwerke, WaveNet und SampleRNN, Tacotron und Transformer-basierte Modelle sowie Transfer Learning. Jede dieser Methoden bietet einzigartige Möglichkeiten und Herausforderungen, und die Kombination mehrerer Methoden kann zu noch fortschrittlicheren TTS-Systemen führen. Die Forschung in diesem Bereich entwickelt sich ständig weiter, und es ist zu erwarten, dass zukünftige Entwicklungen zu noch beeindruckenden Sprachsynthese Ergebnissen führen werden.

## Anwendungen

## Fazit

## Weiterführendes Material

### Podcast
Hier Link zum Podcast.

### Talk
Hier einfach Youtube oder THD System embedden.

### Demo
Hier Link zum Demo Video + Link zum GIT Repository mit dem Demo Code.


## Literaturliste
https://homepages.inf.ed.ac.uk/ckiw/rpml/HMM_speech_synthesis.pdf<br>
https://www.pure.ed.ac.uk/ws/portalfiles/portal/15269212/Speech_Synthesis_Based_on_Hidden_Markov_Models.pdf<br>
https://ieeexplore.ieee.org/abstract/document/7918988<br>
https://www.sciencedirect.com/science/article/abs/pii/S0885230815000200<br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9857677/<br>
https://www.researchgate.net/publication/315696313_Tacotron_A_Fully_End-to-End_Text-To-Speech_Synthesis_Model<br>
https://arxiv.org/pdf/1806.04558.pdf)https://homepages.inf.ed.ac.uk/ckiw/rpml/HMM_speech_synthesis.pdf<br>
https://www.pure.ed.ac.uk/ws/portalfiles/portal/15269212/Speech_Synthesis_Based_on_Hidden_Markov_Models.pdf<br>
https://ieeexplore.ieee.org/abstract/document/7918988<br>
https://www.sciencedirect.com/science/article/abs/pii/S0885230815000200<br>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9857677/<br>
https://www.researchgate.net/publication/315696313_Tacotron_A_Fully_End-to-End_Text-To-Speech_Synthesis_Model<br>
https://arxiv.org/pdf/1806.04558.pdf
