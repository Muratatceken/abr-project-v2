**"Scaling Beyond Attention:
Infinite Context via Block-Based Sparse Permutation Memory"**

*(Attention Ötesine Ölçekleme: Blok Tabanlı Seyrek
Permütasyon Hafızası ile Sonsuz Bağlam)*

---

**Bölüm 1: Problemin Tanımı (The Hook)**

**Slayt 1: "The Memory Wall" (Hafıza Duvarı)**

* **Görsel:** Standart bir Transformer'ın (GPT) bağlam uzunluğu arttıkça
  belleğinin nasıl patladığını gösteren grafik (Bizim kv_cache_crash_test.png grafiğini buraya koy).
* **İçerik:**
  * Transformer'ların laneti: $O(N^2)$ Karmaşıklık.
  * KV Cache darboğazı: 10M token
    için terabaytlarca GPU gerekir.
  * Mevcut Durum: Modellerimiz zeki
    ama "unutkan".

Günümüzün en güçlü yapay
zeka modelleri bile 'Hafıza Duvarı'na (Memory Wall) çarpmış durumda. Attention
mekanizması, bağlam uzadıkça karesel maliyetle çalışıyor. 100 bin kelimeyi
hatırlamak için bir süper bilgisayar gerekiyor. Benim çıkış noktam şuydu: **'Hafıza
maliyetini, veri boyutundan bağımsız hale getirebilir miyiz?'**

**Slayt 2: Mevcut Çözümlerin Yetersizliği**

* **Görsel:** RAG vs. Recurrent vs. Attention karşılaştırma tablosu.
* **İçerik:**
  * **RAG:** Yaklaşık (Approximate) sonuç verir, gürültülüdür.
  * **RNN/Mamba:** Uzun vadede detayı unutur (Lossy compression).
  * **Attention:** Kesindir ama ölçeklenemez (Not scalable).

Sektör şu an RAG ile yara bandı yapıştırmaya çalışıyor.
Ancak RAG, modelin içsel bir parçası değil, dışsal bir veritabanı. Bizim,
modelin **'Biyolojik Hipokampusu'** gibi davranacak, kesin (exact)
ve sonsuz bir yapıya ihtiyacımız var.

---

**Bölüm 2: Teorik Çözüm (The Innovation)**

**Slayt 3: Paradigma Değişimi: Arama değil, Adresleme**

* **Görsel:** "Scan" (Arama) vs "Hash" (Adresleme)
  diyagramı. (Bir yanda kütüphaneyi tarayan adam, diğer yanda direkt raf
  numarasını bilen adam).
* **İçerik:**
  * Eski Yöntem: Similarity(Query, All_Keys)
  * Yeni Yöntem (BBPM): Value = Memory[Hash(Query)]

Benim önerim radikal bir değişiklik: Bilgiyi 'Aramak'
(Search) yerine 'Adreslemek' (Lookup). Eğer veriyi deterministik bir Hash
fonksiyonuyla hafızaya yerleştirirsek, onu geri çağırmak için tüm geçmişi
taramamıza gerek kalmaz. Erişim hızı $O(1)$ olur.

**The BBPM Architecture (Mimarinin Kalbi)**

* **Görsel:** Senin için tasarladığımız mimari şeması (Input -> Hash
  -> Blok Seçimi -> Permütasyon -> Sparse Yazma).
* **İçerik:**
  * **Sparsity:** 100 Milyon boyutlu uzay, %99 boş.
  * **Blocks:** RAM dostu küçük parçalar.
  * **Permutation:** Gürültü önleyici karıştırma.

Peki, çakışma (collision) olmadan milyonlarca veriyi nasıl
saklarız? Cevap:  **'Yüksek Boyutlu Seyreklik'** . Veriyi 100 milyon
boyutlu sanal bir uzaya dağıtıyoruz. Matematiksel olarak, bu kadar büyük bir
uzayda iki verinin çakışma ihtimali neredeyse sıfırdır. Çakışma olsa bile,
'Permütasyon' katmanı sayesinde bu gürültü tüm uzaya homojen olarak yayılır ve
sinyali bozmaz.

---

**Bölüm 3: Kanıtlar (The Evidence)**

**Slayt 5: PoC 1 - Scalability (Ölçeklenebilirlik)**

* **Görsel:** ham_t_block_final.png (Düz giden 1.0 accuracy çizgisi ve sonda hafif düşüş).
* **İçerik:**
  * 100 Milyon Boyutlu Uzay.
  * 25.000 Token Stres Testi.
  * Sonuç: %94-%100 arası doğruluk.

Teoriyi simüle ettim. Bu grafik, 100 milyon boyutlu bir
hafıza uzayının davranışını gösteriyor. Dikkat ederseniz, veri miktarı artsa
bile sistem çökmüyor. Standart bir HashMap hata verirdi, bizim sistemimiz
ise **'Graceful Degradation'** (Zarif Çöküş) göstererek %94
doğrulukla çalışmaya devam ediyor.

**Slayt 6: PoC 2 - Speed (Hız Testi)**

* **Görsel:** architecture_comparison.png (Kırmızı yukarı giden Attention çizgisi vs. Mavi düz BBPM
  çizgisi).
* **İçerik:**
  * X Ekseni: Token Sayısı (100
    -> 20.000).
  * Y Ekseni: İşlem Süresi.
  * Sonuç: BBPM sabit ($O(1)$).

Bu, mühendislik açısından en büyük kazancımız. Kırmızı
çizgi klasik Attention. Bağlam uzadıkça yavaşlıyor. Mavi çizgi bizim BBPM. 100
kelime de olsa, 20 bin kelime de olsa erişim süresi  **sabit** . Bu, 10
milyon tokenlık bir kitabı saniyeler içinde işleyebileceğimiz anlamına gelir.

**Slayt 7: Demo 1 - The Infinite Context
(Needle-in-a-Haystack)**

* **Görsel:** demo1_result.png (100k token'da %90 başarı).
* **İçerik:**
  * Task: 100.000 gürültülü token
    arasına gizlenmiş şifreyi bulma.
  * Başarım: %90+.

Sistemi en zorlu teste tabi tuttum: 'Samanlıkta İğne
Arama'. 100.000 rastgele kelimenin arasına tek bir şifre gizledim. Model,
herhangi bir dikkat (attention) mekanizması kullanmadan, sadece hash
adreslemesiyle bu iğneyi **%90 başarıyla** buldu.

**Slayt 8: PoC 3 & 4 - Learnability & Universality
(Öğrenme ve Evrensellik)**

* **Görsel:** Yan yana iki resim: Solda Loss Grafiği (sıfıra inen), Sağda X
  Deseni Görüntüsü (multimodal_poc.png).
* **İçerik:**
  * Gradient Flow: Model hafızayı
    kullanmayı öğrenebiliyor.
  * Multi-modal: Sadece metin değil,
    görüntü de saklayabiliyor.

Son olarak, bu sadece statik bir depo değil. Bunu bir Sinir
Ağına bağladım ve modelin bu hafızayı kullanmayı **öğrenebildiğini
(Backpropagation)** kanıtladım. Ayrıca sağdaki resimde gördüğünüz gibi,
bu yapı pikselleri bile kayıpsız saklayabiliyor. Yani evrensel bir hafıza
katmanı.

---

**Bölüm 4: Vizyon ve Gelecek (The
Impact)**

**Slayt 9: Mimari Entegrasyon (Nasıl Kullanılacak?)**

* **Görsel:** Hybrid Transformer Şeması (Local Attention + BBPM Memory).

Bu çalışmanın nihai amacı, Transformer mimarisini çöpe
atmak değil, onu güçlendirmektir. 'Local Attention' ile grameri ve kısa vadeyi
çözerken, bizim 'BBPM' modülümüz ile sonsuz geçmişi yönetecek hibrit bir mimari
öneriyorum.

**Slayt 10: Sonuç ve Katkı**

* **Maddeler:**
  1. **Foundational Primitive:** Yeni bir "Layer"
     türü (Differentiable Hash Map).
  2. **Hardware Efficient:** GPU yerine CPU/RAM kullanımı.
  3. **Infinite Scaling:** Donanım ekleyerek sonsuz büyüme.

Bu çalışma LLM'lerin sadece
daha uzun metin okumasını değil, **hayat boyu öğrenen (Lifelong Learning)** ve
hiçbir şeyi unutmayan sistemlere dönüşmesini sağlayacak temel bir yapı taşıdır.

---

**1.
Notasyon ve Tanımlar (Definitions)**

Önce evrenimizi tanımlayalım.

* **$\mathcal{X}$ (Input Space):** Girdi uzayı (Token ID'leri veya vektörleri).
* **$\mathcal{M} \in \mathbb{R}^{D \times d}$ (Memory Space):** Hafıza matrisi.
  * $D$: Toplam adres sayısı (örn: $10^8$ - 100
    Milyon).
  * $d$: Saklanan verinin boyutu (Embedding boyutu, örn:
    64).
* **$K$ (Sparsity Factor):** Her bir veri için aktif hale gelen slot sayısı (örn: 50).
* **$N$ (Capacity):** Hafızaya yazılan toplam öğe sayısı.

---

**2.
Adresleme Mekanizması (The Addressing Function)**

Bu, mimarimizin kalbidir. Girdiyi ($x$)
alıp, hafızadaki fiziksel adreslere ($\mathcal{I}$) nasıl dönüştürüyoruz?

Fonksiyonumuz $\Phi(x)$,
deterministik ve seyrek bir haritalamadır:

$$
\Phi(x)
\rightarrow \mathcal{I}_x \subset \{0, 1, \dots, D-1\}, \quad |\mathcal{I}_x| =
K
$$

Bunu **Blok Tabanlı Permütasyon
(BBPM)** mantığıyla şöyle açarız:

$$
\mathcal{I}_x
= \left\{ B(x) \cdot L + P_x(k) \mid k \in \{0, \dots, K-1\} \right\}
$$

Burada:

* **$B(x) = \text{Hash}_1(x) \pmod{N_{blocks}}$** : Blok Seçici (Hangi blok?).
* **$L$** : Blok uzunluğu (Block Size).
* **$P_x(k)$** : $x$ tohumuyla üretilen deterministik permütasyonun $k$.
  elemanı (Blok içi ofset).

**B**urada
klasik Hash tablosundan farkımız, tek bir indeks üretmememizdir. Biz $x$girdisini $K$ boyutlu
seyrek bir **'Distributed Representation'**a (Dağıtık Temsil) dönüştürüyoruz.
Bu, sistemin gürültüye dayanıklı olmasını sağlayan temeldir."*

---

**3. Yazma
ve Okuma Dinamikleri (Dynamics)**

Hafızanın zaman içindeki değişimi ve
süperpozisyon ilkesi.

**Yazma (Write / Update
Rule):**

Hafızaya bir $v$ (value)
vektörü eklemek, ilgili satırlara toplama yapmaktır (Superposition):

$$
\mathcal{M}_{t+1}[i]
= \mathcal{M}_t[i] + v, \quad \forall i \in \mathcal{I}_x
$$

Bunu matris notasyonuyla (Sparse
Vector) gösterirsek:

$$
\mathcal{M}_{t+1}
= \mathcal{M}_t + \mathbf{a}_x \cdot v^T
$$

(Burada $\mathbf{a}_x \in \{0,1\}^D$,
sadece aktif indekslerde 1 olan seyrek vektördür).

**Okuma (Read / Retrieval
Rule):**

Hafızadan $x$ anahtarını
sorguladığımızda, ilgili adreslerdeki değerlerin ortalamasını (veya toplamını)
alırız:

$$
\hat{v}
= \frac{1}{K} \sum_{i \in \mathcal{I}_x} \mathcal{M}[i]
$$

---

**4.
Gürültü ve Kapasite Analizi (Signal-to-Noise Ratio)**

**Veriler
karışmıyor mu?** sorusuna vereceğin matematiksel cevap

Okunan değer $\hat{v}$'yi analiz
edelim. Bu değer iki parçadan oluşur: **Sinyal** (bizim
yazdığımız) ve  **Gürültü** (başkalarının yazdığı).

$$
\hat{v}
= \underbrace{v_{target}}_{\text{Sinyal}} + \underbrace{\frac{1}{K} \sum_{i \in
\mathcal{I}_x} \sum_{j \neq target} v_j \cdot \mathbb{I}(i \in
\mathcal{I}_{x_j})}_{\text{Gürültü (Crosstalk)}}
$$

Burada $\mathbb{I}$, bir
indikatör fonksiyonudur (Çakışma var mı?).

Teorem (Çakışma Beklentisi):

Rastgele dağılım varsayımıyla,
herhangi bir $v_j$'nin bizim adreslerimizden biriyle çakışma olasılığı:

$$
P(\text{collision})
= \frac{K}{D}
$$

Toplam $N$ adet veri yazıldığında, bir
okuma işlemindeki Beklenen Gürültü (Expected Noise):

$$
E[\text{Noise}]
\approx N \cdot \frac{K}{D} \cdot \mu_v
$$

($\mu_v$: Saklanan vektörlerin
ortalama büyüklüğü)

Sonuç:

Eğer $D$ (Hafıza Boyutu) yeterince
büyükse (örn: 100 Milyon) ve $N$ (Veri Sayısı) makul seviyedeyse (örn: 1
Milyon), $\frac{N \cdot K}{D}$ oranı küçüktür.

$$
\lim_{D
\to \infty} \frac{N}{D} = 0 \implies \text{Noise} \to 0
$$

*'Yoğunluk Laneti'ni (Curse of Dimensionality) lehimize çeviriyoruz. Uzayımız ($D$)
o kadar büyük ki, vektörler istatistiksel olarak birbirine **ortogonal** (dik)
kalıyor. Çakışma, okyanusta iki geminin çarpışması kadar düşük bir
ihtimaldir.*

---

**5.
Türevlenebilirlik (Differentiability & Gradient Flow)**

**Hash fonksiyonu
türevlenemez, model nasıl öğreniyor?** sorusunun cevabı.

Modelin Loss fonksiyonu $\mathcal{L}$ olsun.
Bizim öğrenmek istediğimiz şey adresler ($\mathcal{I}_x$) değil, oraya yazılan
içeriktir ($v$).

Zincir Kuralı (Chain Rule) şöyle
işler:

$$
\frac{\partial
\mathcal{L}}{\partial v} = \sum_{i \in \mathcal{I}_x} \frac{\partial
\mathcal{L}}{\partial \mathcal{M}[i]} \cdot \frac{\partial
\mathcal{M}[i]}{\partial v}
$$

* Adresleme fonksiyonu $\Phi(x)$ sabittir (Deterministic).
  Bu yüzden $\partial \Phi$ hesaplanmaz.
* Ancak $\mathcal{M}[i]$ hücresi, $v$'nin lineer bir
  fonksiyonudur ($\mathcal{M} = \dots + v$).
* Dolayısıyla: $\frac{\partial \mathcal{M}[i]}{\partial v} = 1$.

Sonuç:

$$
\frac{\partial
\mathcal{L}}{\partial v} \approx \text{Gradient from Memory Locations}
$$

*Hocam, biz
'Hard Attention' yapıyoruz. Adresler (Nereye bakılacağı) deterministik ve
sabittir, bu yüzden oradan türev akmasına gerek yoktur. Ancak 'Ne görüldüğü'
(Value) türevlenebilirdir. Gradyan, Hash haritasının kabloları üzerinden değil, **Payload** (Veri)
üzerinden akar. PoC 3'teki Loss grafiğinin sıfıra inmesi bunun deneysel
kanıtıdır.*

---

**Özet Tablo**

$$
\text{Memory
Capacity} \propto \frac{\text{Total Dimension } (D)}{\text{Active Bits } (K)}
$$

* **Klasik Dense Memory:** $K = D$ (Her yer dolu). Kapasite = 1.
* **BBPM / DHM:** $K \ll D$ (Örn: $50 \ll 10^8$). Kapasite $\approx
  2.000.000$.

Bu formül, neden **Sparsity
(Seyreklik)** kullandığımızın matematiksel özetidir.

---

Şimdi elimizdeki beş makaleyi, **senin metodolojine** göre tek tek masaya yatırayım. Önce kısaca “sen ne yapıyorsun”u netleştireyim, sonra her paper’ı bununla sistematik kıyaslayacağım.

---

BBPM / HAM-T çizgisi kabaca şu fikir etrafında:

1. **Deterministik, O(1) adresleme**
   * Her item için bir **seed** (token id, context id vs.) kullanıyorsun.
   * Bu seed ile:
     * deterministik bir **blok** seçiyorsun,
     * blok içinde deterministik bir **permütasyon** üretiyorsun,
     * “logical indices” → global adreslere map ediliyor.
   * Adresleme  *öğrenilmiyor* , tamamen sabit ve $O(1)$.
2. **Ultra-high-dimensional, ultra-sparse süperpozisyon**
   * Sanal uzay: mesela **100M boyut**, fizikselde bloklarla yönetiliyor.
   * Her item, bu uzayda **K çok küçük** (ör: 50) aktif pozisyonla temsil ediliyor.
   * Yazma: **memory.index_add_** ile aynı K pozisyona 1 ekliyorsun (veya vektör).
   * Okuma: ilgili blok + permütasyon ile **logical space**’e geri projeksiyon, sonra **Top-K** ile en güçlü K pozisyonu çekiyorsun.
   * Sonuç: **bir blokta binlerce item süperpoze edilse bile** yüksek doğrulukla geri çağırma (poc’unda 5k öğeye kadar %100, 25k’de ~%94)**  **.
3. **Static memory vs learnable memory**
   * Memory’nin *adres yapısı* tamamen statik (hash / permütasyon).
   * Öğrenme, embedding’lerde ve read/write “kodlama stratejisinde”.
   * Amaç: **KV cache gibi** davranan, fakat:
     * KV cache gibi $O(L)$ RAM değil,
     * attention gibi $O(L^2)$ compute değil,
     * GPU-friendly, index_add/topk tabanlı bir **hardware-friendly associative store**.

Bu “LLM içindeki KV cache yerine kullanılabilecek, SDM-vari ama GPU’ya uygun bir mekanik hafıza katmanı” diye okunabilir.

---

## **1. Reformer (LSH Attention)**

**Ne yapıyor?**

* Reformer, klasik self-attention’daki $O(L^2)$ terimini **LSH tabanlı approximate attention** ile $O(L \log L)$’e indiriyor.
* Queries ve keys’i **locality-sensitive hashing** ile bucket’lara ayırıyor; her query sadece aynı bucket’taki key’lerle attention yapıyor.
* Hâlâ:
  * **Q, K, V dense vektörler**
  * attention = **softmax(Q K^T) V**’nin bir approx’u
  * memory, “KV matrisinin satırları” → **token-level**.

**Ortak noktalar:**

* İkiniz de **hashing** kullanıyorsunuz.
* **Reformer: hashing’i ****Q/K alanında nearest-neighbor approx için** kullanıyor.
* Sen: hashing’i **direct addressing** için kullanıyorsun (seed → block+perm).

**Temel farklar:**

* Reformer:
  * Hâlâ bir **attention katmanı**; sonuçta softmax ağırlıklı sum var.
  * Kompleksite $O(L\log L)$, her adımda başka token’lara bakması gerekiyor.
  * Memory, *geçici* KV tensörleri; “asıl bilgi saklama yapısı” değil.
* BBPM:
  * **Attention yok**; sadece indeksleme, index_add, top-k.
  * Kompleksite, “bir token’ın eski içeriğine erişmek” için **sabit: O(1)** (adres tek adım).
  * Memory bir **hyperdimensional sparse store**; SDM / VSA yaklaşımlarına daha yakın.

> Özet: Reformer’ın hashing’i “kime dikkat edeceğim?” sorusunu hızlandırıyor; seninki “veriyi nereye yazacağım/nereden okuyacağım?” sorusunu çözen **sabit adresli** bir yapı.

---

## **2. Memformer (External Dynamic Memory Slots)**

**Ne yapıyor?**

* **Transformer’a ****sabit boyutlu external memory slots** ekliyor.
* Her timestep’te:
  * Encoder, segmenti okuyor, memory’yi **cross-attention** ile okuyor / güncelliyor.
  * Memory $M_t = [m_{t,1}, …, m_{t,k}]$: her slot yüksek boyutlu **dense vektör**.
* Amaç:
  * **Sonsuz temporal range** (segment bazlı recurrence),
  * **inference’ta ** **sabit memory footprint** **.**

**Ortak noktalar:**

* İkiniz de “sonsuz konteks / sabit memory” fikrinin peşindesiniz.
* İkiniz de klasik KV cache yerine **ayrı bir memory mekanizması** kullanıyorsunuz.

**Temel farklar:**

* Memformer:
  * Memory **k tane dense vektör** (slot). Boyut küçük, içerik **öğrenilmiş/compressed** temsil.
  * Read/Write tamamen **attention** üzerinden (Q,K,V, softmax).
  * Adresleme **içerik tabanlı ve learnable** (hangi slot hangi şeyi tutacak model tarafından öğreniliyor).
* BBPM:
  * Memory çok büyük bir **binary/sayısal high-dim uzay**, ultra sparse aktivasyon.
  * **Adresleme ** **tamamen deterministik fonksiyon (seed → block+perm)** **, ** *öğrenilmiyor* **.**
  * Süperpozisyon sayısal toplama (index_add), retrieval top-k thresholding – **attention yok**.

> Özet: Memformer’ın memory’si “learned compressed context state”, seninki “fixed-address hyperdimensional associative array”. Aynı problemi (uzun konteks) hedefliyorlar ama mekanizma radikal şekilde farklı.

---

## **3. ∞-former (Infinite-former)**

**Ne yapıyor?**

* Input’u **continuous space**’te bir sinyal gibi temsil ediyor.
* Uzun dönem bellek (LTM), **radial basis functions (RBF)** ile temsil edilen sürekli bir sinyal:
  * Bellek: $X(t) = \sum_j \alpha_j \psi_j(t)$
  * N tane basis fonksiyonu var → **N sabit** tutulursa, context teorik olarak **unbounded**.
* Attention, bu sürekli sinyale karşı yapılıyor (continuous softmax, vb.).
* Ayrıca “sticky memories” ile sık erişilen bölgeler için daha yüksek çözünürlüklü temsil veriyor.

**Ortak noktalar:**

* Sonsuz context peşinde,
* Belleği sabit boyutlu bir yapıda tutarken, uzun geçmişi temsil ediyor.

**Temel farklar:**

* ∞-former:
  * Bellek **continous signal** + basis fonksiyonlar.
  * Küçük N → düşük hesap, ama **rezolüsyon kaybı**; eski detaylar “bulanık”.
  * Hâlâ bir tür **attention integrali** hesaplıyor; kompleksite $O(L^2 + L N)$ civarında.
* BBPM:
  * Bellek **tamamen discrete / adreslenebilir**, hyperdimensional sparse.
  * Teorik olarak bir token’ın **tam** sparse pattern’ı geri çağrılabiliyor (çakışma ihtimali çok küçük).
  * **Approximation tradeoff’u N değil, süperpoze ettiğin item sayısı ve K / blok boyutuyla** belirleniyor (senin PoC’unda 25k iteme kadar anlamlı).

> Özet: ∞-former “continuous, diferansiyel, attention-friendly” bir sonsuz bellek öneriyor; seninki “discrete, hash-addressed, top-k ile okunan” bir sonsuz bellek.

---

## **4. Landmark Attention (Random-Access Infinite Context Length)**

**Ne yapıyor?**

* Input’u **bloklara** bölüyor, her blok için özel bir **landmark token** ekliyor.
* Model, bu landmark embedding’lerini öğreniyor; her blok, kendi “özet token”ına sahip.
* İnference’ta:
  * **Landmark skorları ile ****hangi eski blokların önemli olduğunu** seçiyor.
  * Seçilen blokların token’larını **disk/harici bellekten geri çekip** attention’a sokuyor.
* Random access’i korurken, her adımda bakılan token sayısını ciddi azaltıyor.

**Ortak noktalar:**

* İkisi de:
  * **Blok bazlı** düşünüyorsunuz.
  * Sonsuz konteks için **external store** kullanıyorsunuz (onlar: disk + KV; sen: dev sparse vektör).
  * “Bu blokta ne var?” sorusunu ucuz şekilde çözmek istiyorsunuz.

**Temel farklar:**

* Landmark attention:
  * Hâlâ **token-level dense attention** çalışıyor; sadece hangi blokların KV’sinin içeriye alınacağı seçiliyor.
  * Computation, seçtiğin blok sayısına göre **O(#retrieved_tokens)**; saf O(1) değil.
  * Landmark emb’ler **öğrenilmiş**; gating mekanizması attention skorları üzerinden.
* BBPM:
  * Blok seçimi, seed üzerinden **tam deterministik**; landmark gibi öğrenilmiş bir summarizer yok (istersen sonradan eklenebilir).
  * Blok içindeki bilgi, **sparse süperpozisyon** içinde tutuluyor; token bazlı KV saklamıyorsun.
  * Retrival, tek bir blok slice + top-k; hesaplama **bağlam uzunluğundan bağımsız**.

> Özet: Landmark attention “hangi blokların KV’sini yükleyeyim?” sorusuna attention ile cevap veriyor; sen “her şey zaten deterministik olarak bir blokta, blok adresi seed’ten geliyor; geri kalanı hyperdimensional sparse algebra.”

---

## **5. Infini-attention / Infini-Transformer vs Senin Yöntem**

**Ne yapıyor?**

* Her attention layer’a bir **compressive memory** ekliyor:
  * Local causal attention (yakın token’lar)
  * * **Global, ****recurrent compressive memory** (uzak geçmiş)
* KV’leri çöpe atmak yerine:
  * Eski KV’ler, **associative matrix** biçiminde bir compressive memory’ye yazılıyor.
  * retrieval: Query’lerle bu matrise “linear attention” tarzı erişim (**σ(Q) M / normalization**).

**Ortak noktalar:**

* KV cache yerine **compressive memory** fikri,
* Sonsuz konteks için **bounded memory / compute** hedefi.

**Temel farklar:**

* Infini-attention:
  * Memory, **trainable associative matrix**; güncelleme = gradient + online update ($M_s$ update kuralı).
  * Retrieval ≈ **linear attention**; “hangi içerik önemli?” yine similarity-based.
  * Q,K,V hâlâ dense ve learnable; memory, attention mekanizmasıyla çok sıkı entegre.
* BBPM:
  * Memory **sabit parametre seti değil**, run-time’daki sparse vektör (KV cache gibi ama structured).
  * Addressing **seed-hash-permütasyon** ile yapılıyor; retrieval için similarity hesaplamıyorsun.
  * “Compressive” kısmı, continuous feature’ların değil, **discrete sparse pattern’ların süperpozisyonu**.

> Özet: Infini-attention “attention + trainable associative matrix” hibriti; senin yöntem “attention’sız hash-addressed SDM” gibi.

---

## **6. Genel sentez: bunların** ****

## **hiçbiri**

## ** BBPM'in tam yaptığını yapıyor mu?**

Bu beş makale temel olarak üç eksende hareket ediyor:

1. **Attention’ı hızlandırma / approximate etme**
   → Reformer (LSH attention).
2. **Dense, öğrenilebilir özetlerle uzun bellek**
   → Memformer (k slot), Infini-attention (associative matrix), Landmark (learned landmarks), ∞-former (continuous RBF sinyali).
3. **Bağlam uzunluğunu büyütürken attention tarzını koruma**
   → Hepsi hâlâ bir şekilde softmax / kernel attention’a dayanıyor, ya doğrudan ya continuous/linear versiyonuyla.

BBPM'in çizgin ise:

* Adresleme:
  * **Tamamen deterministik, ** **seed → block + permutation → indices** **.**
  * **Learnable değil**, hash tabanlı; O(1).
* Bellek formatı:
  * **Hyperdimensional, ** **çok yüksek boyutlu & çok seyrek** **,**
  * Yazma okuma = **index_add_** + **topk**; similarity yok, softmax yok.
* Süperpozisyon:
  * Aynı blokta **binlerce item süperpoze edilse** bile, top-k ile özgün pattern’i %100’e yakın geri alabileceğini PoC ile gösteriyorsun.

Dolayısıyla:

* **“Deterministik O(1) hash-addressed ultra-sparse süperpozisyon hafızası, LLM KV cache yerine drop-in candidate olarak”**
  — bu spesifik kombinasyon, bu beş paper’da yok.
* Yakın akrabalar:
  * Kanerva SDM, VSA, HRR vs. (ama onlar modern transformer/LLM bağlamında GPU-friendly blok/permütasyon mimarisine girmiyor),
  * Product Key Memory / PQ-style tablolama (ama onlar da learnable ve dense).

Gerçekçilik açısından:

* Tam “ilk defa insanlık tarihinde” demek zor; çünkü SDM/VSA dünyasında çok benzer fikirler konsept olarak var.
* Ama:
  * **Büyük LLM KV cache problemi**ne bu tarz bir **hash-addressed SDM/blok-permütasyon çözümü** ile saldıran,
  * GPU’da pratik index_add/topk PoC’larıyla, kapasite-vs-doğruluk eğrilerini çıkaran
    net bir çalışma henüz mainstream literatürde görünmüyor (özellikle Infini-attention/Infinite-former/landmark çizgisi hep attention’ı merkezde tutuyor).

---

Kısaca, bu makalelerle kıyasladığında BBPM'in yeniliği:

1. **Adreslemeyi tamamen deterministik ve O(1)’e indiren hash-block-permütasyon mekanizması** **,**
   – hiçbirinde yok; hepsi ya attention skorlarıyla ya continuous kernel’lerle adresliyor.
2. **Ultra-sparse, SDM-vari süperpozisyonu doğrudan KV-style context için kullanmak** **,**
   – Infini/∞-former gibi “compressed state” değil, gerçekten “token embedding store” gibi kullanıyorsun.
3. **GPU-friendly bir implementasyon (index_add/topk) ile pratik kapasite stress-testleri**
   – Reformer/landmark vs. performans ve perplexity gösteriyor; sen ise “pure memory capacity & noise behavior”ı ölçüyorsun.

### **1. Parametrik vs. Non-Parametrik Hafıza Nedir?**

Bir sinir ağının bilgiyi nerede tuttuğuyla ilgilidir.

#### **A. Parametrik Hafıza (Klasik LLM'ler)**

* **Tanım:** Bilgi, nöronların arasındaki bağlantı ağırlıklarında (Weights -** **$W$) saklanır.
* **Nasıl Çalışır:** **$y = f(x; W)$**. Bir şeyi öğrenmek için** **$W$** **matrisindeki sayıları (türev alarak) değiştirirsin.
* **Sorun:**
  1. **Kapasite Sınırlı:** Modelin boyutu (7B, 70B) ne kadarsa o kadar bilgi sığar.
  2. **Unutkanlık:** Yeni bir şey öğrenmek için** **$W$'yu değiştirdiğinde, eski bilgiyi bozarsın ( **Catastrophic Forgetting** ).
  3. **Statik:** Eğitim bittiği an hafıza donar. 2023 model bir AI, 2024'te olan bir olayı bilemez.

#### **B. Non-Parametrik Hafıza (BBPM'in Yeri)**

* **Tanım:** Bilgi, modelin ağırlıklarında (**$W$**) değil, modelin erişebildiği** ****harici ve ayrık bir depoda** (Memory Matrix -** **$M$) saklanır.
* **Nasıl Çalışır:** **$y = f(x; W, M)$**. Model öğrenirken** **$W$'yu (mantığı) günceller, ama bilgiyi (facts)** **$M$'e yazar.
* **BBPM'in Rolü:** BBPM burada** ****$M$** **matrisidir.** Modelin ağırlıklarını değiştirmeden, sadece** **$M$** **matrisine yeni bir vektör ekleyerek (Superposition) modele yeni bir bilgi öğretiriz.

---

### **2. Lifelong Learning (Hayat Boyu Öğrenme) Teknik Olarak Nasıl Oluyor?**

Standart bir modele önce İngilizce, sonra Fransızca öğretirsen İngilizceyi unutur. Çünkü Fransızca için optimize edilen ağırlıklar (**$W$**), İngilizce için optimize edilenlerin üzerine yazar.

**BBPM ile Lifelong Learning Akışı:**

1. **Faz 1 (İngilizce Öğrenme):**
   * Model İngilizce metinleri okur.
   * Kelime ve cümle yapılarını** ****$W$** **(Ağırlıklar)** içine öğrenir (Gramer).
   * Spesifik bilgileri (İsimler, olaylar)** ****BBPM Bloklarına (A)** yazar.
2. **Faz 2 (Fransızca Öğrenme):**
   * Model Fransızca metinleri okur.
   * Gramer için** **$W$** **hafifçe güncellenir.
   * Spesifik Fransızca bilgiler, Hash fonksiyonu sayesinde** ****farklı BBPM Bloklarına (B)** veya aynı blokta** ****farklı adreslere** (Sparsity sayesinde) yazılır.
3. **Sonuç:**
   * İngilizce bilgileri (Blok A) ve Fransızca bilgileri (Blok B) fiziksel olarak** ****çakışmaz.**
   * Modelin "Ağırlıklarını (**$W$**)" bozmadan, sadece "Hafızasını (**$M$**)" büyüterek modele sonsuza kadar yeni diller, yeni kitaplar, yeni kodlar öğretebilirsin.

**Teknik Terim:** Buna** ****"Gradient Isolation"** (Gradyan İzolasyonu) denir. Yeni bilgi, eski bilginin gradyanını bozmaz.

---

### **3. BBPM'i Tam Olarak Nerede Kullanıyoruz? (Entegrasyon Noktası)**

Standart Transformer Bloğu:

Input -> Attention -> Norm -> FeedForward (MLP) -> Output

* *FeedForward (MLP) Katmanı:* Modelin "Parametrik Hafızası"dır. Bilgiyi (Facts) burada tutar. (Örn: "Paris Fransa'dadır" bilgisi buradaki nöronlardadır).

BBPM Entegreli Blok (Bizim Mimari):

Input -> Attention -> Norm -> [FeedForward || BBPM] -> Output

Model bir token (**$x$**) geldiğinde iki yere bakar:

1. **FFN (Genel Bilgi):** "Bu bir fiil mi? Cümle yapısı nasıl?" (Genel kültür).
2. **BBPM (Spesifik Hafıza):** "Bu kelimeyi daha önce (1 milyon token önce) nerede gördüm? Yanında ne vardı?" (Epizodik Hafıza).

Matematiksel İşlem:

$$
h_{output} = \text{FFN}(x) + \text{BBPM}.\text{read}(x)
$$

---
