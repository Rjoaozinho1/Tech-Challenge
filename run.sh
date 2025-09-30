python train_fn.py \
  --data_path trn.clean.jsonl \
  --model_name google/flan-t5-base \
  --output_dir ./models/flan_t5_a10_lora \
  --peft true \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --load_in_8bit true \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --max_source_length 256 \
  --max_target_length 256 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --run_baseline true \
  --do_eval true \
  --merge_lora true

---------------------------------------
python train_fn.py \
  --infer "Qual é a descrição do produto 'iPhone 12 128GB'?" \
  --use_trained ./models/flan_t5_a10_lora

---------------------------------------
[load] base=google/flan-t5-base
[load] adapter from ./models/flan_t5_a10_lora
[device] Using cuda
[samples] Generating 3 fine-tuned predictions

=== Sample 1 ===
Prompt: Pergunta: Qual é a descrição do produto "Paris on the Eve 1900-1914"?
Contexto: título = Paris on the Eve 1900-1914
Responda apenas com a descrição.
Target: In pre-WW I Paris, Picasso, Debussy, Gide, Proust, Henri Bergson and Pierre and Marie Curie were among the creative minds who helped forge the modern world view of a subjective, fluid reality. Cronin's achievement in this scintillating, highly enjoyable social and cultural history is to demonstrate how the various endeavors of these and other groundbreakers were interrelated. France's adaptation to unless France really developed it.aa the motor car changed Proust's way of life. A new subjectivism, spurred by Bergson's spiritual philosophy, made possible Debussy's exploration of musical nuance, Picasso and Braque's invention of Cubism, and poet Charles Peguy's vision of a socialist utopia. Mingling gossip, biography and astute commentary, this chronicle tracks Diaghilev, Colette, Mattise and a host of others. Cronin, biographer of Napoleon, argues persuasively that Parisians' chilly rancor toward Germans stemmed in part from France's sense of national pride. Do you mean: 'French national pride derived from winning the Franco-German rivalry over imperialist expansion in Africa.'?aa Photos.Copyright 1991 Reed Business Information, Inc.--This text refers to an out of print or unavailable edition of this title.
Fine-tuned: "An excellent book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It is a great book for anyone who loves the Paris of the Eve. It

=== Sample 2 ===
Prompt: Pergunta: Qual é a descrição do produto "California, The Beautiful Cookbook: Authentic Recipes from California"?
Contexto: título = California, The Beautiful Cookbook: Authentic Recipes from California
Responda apenas com a descrição.
Target: Two collections of California's best. The latest addition to Collins's popular series ( America the Beautiful Cookbook, LJ 11/15/90; France the Beautiful Cookbook, LJ 10/15/89; et al.) is a locally specialized but stunning entry. The photographs of the California scenery/scene are varied and vivid, while the food photography is quite spectacular, and the recipes are far better and more interesting than those in several of the previous books. Recommended for most California libraries, and for any others that can afford this luxury. Like Robert Schneider's The Open Hand Cookbook ( LJ 11/15/89), the San Francisco Symphony's benefit cookbook offers a tantalizing collection of menus from some of California's most talented chefs. The roster of Bay Area contributors includes Jeremiah Tower, Alice Waters, Barbara Tropp, Ken Hom, and many other stars; many of the recipes are unmistakably "California," but they do in fact represent a wide range of styles, and they've been well edited for the home cook. Recommended.Copyright 1991 Reed Business Information, Inc.
Fine-tuned: "It's a wonderful book. It's a great way to learn about the state and its people."--Joseph L. Sullivan, author ofThe Beautiful Cookbook: California, The Beautiful Cookbook"It's a great book. It's a great way to learn about the state and its people."--Joseph L. Sullivan, author ofThe Beautiful Cookbook: California, The Beautiful Cookbook"It's a great book. It's a great way to learn about the state and its people."--Joseph L. Sullivan, author ofThe Beautiful Cookbook: California, The Beautiful Cookbook"It's a great book. It's a great way to learn about the state and its people."--Joseph L. Sullivan, author ofThe Beautiful Cookbook: California, The Beautiful Cookbook"It's a great book. It's a great way to learn about the state and its people."--Joseph L. Sullivan, author ofThe Beautiful Cookbook: California,

=== Sample 3 ===
Prompt: Pergunta: Qual é a descrição do produto "Italy, The Beautiful Cookbook: Authentic Recipes from the Regions of Italy"?
Contexto: título = Italy, The Beautiful Cookbook: Authentic Recipes from the Regions of Italy
Responda apenas com a descrição.
Target: Lorenza de Medici has published more than 30 cookbooks. She has appeared in a 13-part series on Italian cooking for public television and conducts a cooking school at Badia a Coltibuono, an 11th-century estate and winery near the Chianti region of Tuscany. She divides her time between Milan and Badia a Coltibuono.
Fine-tuned: "Italy, The Beautiful Cookbookis a wonderful book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food. It is a great book for anyone who loves Italian food.

[load] model=google/flan-t5-base
[device] Using cuda
[samples] Generating 3 baseline predictions

=== Sample 1 ===
Prompt: Pergunta: Qual é a descrição do produto "Paris on the Eve 1900-1914"?
Contexto: título = Paris on the Eve 1900-1914
Responda apenas com a descrição.
Target: In pre-WW I Paris, Picasso, Debussy, Gide, Proust, Henri Bergson and Pierre and Marie Curie were among the creative minds who helped forge the modern world view of a subjective, fluid reality. Cronin's achievement in this scintillating, highly enjoyable social and cultural history is to demonstrate how the various endeavors of these and other groundbreakers were interrelated. France's adaptation to unless France really developed it.aa the motor car changed Proust's way of life. A new subjectivism, spurred by Bergson's spiritual philosophy, made possible Debussy's exploration of musical nuance, Picasso and Braque's invention of Cubism, and poet Charles Peguy's vision of a socialist utopia. Mingling gossip, biography and astute commentary, this chronicle tracks Diaghilev, Colette, Mattise and a host of others. Cronin, biographer of Napoleon, argues persuasively that Parisians' chilly rancor toward Germans stemmed in part from France's sense of national pride. Do you mean: 'French national pride derived from winning the Franco-German rivalry over imperialist expansion in Africa.'?aa Photos.Copyright 1991 Reed Business Information, Inc.--This text refers to an out of print or unavailable edition of this title.
Baseline: "Paris on the Eve 1900-1914"

=== Sample 2 ===
Prompt: Pergunta: Qual é a descrição do produto "California, The Beautiful Cookbook: Authentic Recipes from California"?
Contexto: título = California, The Beautiful Cookbook: Authentic Recipes from California
Responda apenas com a descrição.
Target: Two collections of California's best. The latest addition to Collins's popular series ( America the Beautiful Cookbook, LJ 11/15/90; France the Beautiful Cookbook, LJ 10/15/89; et al.) is a locally specialized but stunning entry. The photographs of the California scenery/scene are varied and vivid, while the food photography is quite spectacular, and the recipes are far better and more interesting than those in several of the previous books. Recommended for most California libraries, and for any others that can afford this luxury. Like Robert Schneider's The Open Hand Cookbook ( LJ 11/15/89), the San Francisco Symphony's benefit cookbook offers a tantalizing collection of menus from some of California's most talented chefs. The roster of Bay Area contributors includes Jeremiah Tower, Alice Waters, Barbara Tropp, Ken Hom, and many other stars; many of the recipes are unmistakably "California," but they do in fact represent a wide range of styles, and they've been well edited for the home cook. Recommended.Copyright 1991 Reed Business Information, Inc.
Baseline: "California, The Beautiful Cookbook: Authentic Recipes from California"

=== Sample 3 ===
Prompt: Pergunta: Qual é a descrição do produto "Italy, The Beautiful Cookbook: Authentic Recipes from the Regions of Italy"?
Contexto: título = Italy, The Beautiful Cookbook: Authentic Recipes from the Regions of Italy
Responda apenas com a descrição.
Target: Lorenza de Medici has published more than 30 cookbooks. She has appeared in a 13-part series on Italian cooking for public television and conducts a cooking school at Badia a Coltibuono, an 11th-century estate and winery near the Chianti region of Tuscany. She divides her time between Milan and Badia a Coltibuono.
Baseline: "Italy, The Beautiful Cookbook: Authentic Recipes from the Regions of Italy"