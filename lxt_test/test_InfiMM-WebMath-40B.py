# 本文件用于抽取 /mnt/hwfile/llm-safety/datasets/InfiMM-WebMath-40B 内的一部分数据来调用大模型生成思维链回答，并用另一个大模型来提取答案，然后对比参考答案判断是否正确，将抽取的数据、模型生成的回复、提取的答案、正确性判定的结果都保存到新的json文件中。

import pandas as pd
import json

# 读取parquet文件
parquet_path = '/mnt/hwfile/llm-safety/datasets/InfiMM-WebMath-40B/part-00000-d02d03ca-9d67-4b41-ae48-a1ec4116ea42-c000.gz.parquet'
df = pd.read_parquet(parquet_path)

# 打印数据集基本信息
print("Dataset Info:")
print(df.info())
print("\nDataset Columns:")
print(df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))

# 打印一个具体样本的详细内容
print("\nDetailed content of first sample:")
first_sample = df.iloc[0]
for col in df.columns:
    print(f"\n{col}:")
    print(first_sample[col])

# 结果为
# └─$ python lxt_test/test_InfiMM-WebMath-40B.py 
# Dataset Info:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 59374 entries, 0 to 59373
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype 
# ---  ------      --------------  ----- 
#  0   URL         59374 non-null  object
#  1   text_list   59374 non-null  object
#  2   image_list  59374 non-null  object
#  3   metadata    59374 non-null  object
# dtypes: object(4)
# memory usage: 1.8+ MB
# None

# Dataset Columns:
# ['URL', 'text_list', 'image_list', 'metadata']

# First 3 rows:
#                                                  URL  ...                                           metadata
# 0                        https://lokobar.pl/?p=30023  ...  {"ft_lang_label":"__label__en","ft_lang_prob":...
# 1  https://www.nneellinvest.com/post/average-dire...  ...  {"ft_lang_label":"__label__en","ft_lang_prob":...
# 2  https://www.pndhs.org/course/topics-in-college...  ...  {"ft_lang_label":"__label__en","ft_lang_prob":...

# [3 rows x 4 columns]

# Detailed content of first sample:

# URL:
# https://lokobar.pl/?p=30023

# text_list:
# ['Request for Quotation\n\nYou can get the price list and a A&C representative will contact you within one business day.('
#  None
#  '[email\xa0protected])\n\nmotor power calculation for conveyor xls Description\n\n•'
#  None
#  'Conveyor Horsepower Calculator - Superior …\n\nConveyor Horsepower Calculator Sara Hoidahl 2017-08-28T21:17:11+00:00. Conveyor Length (center-to-center) Belt Width. Vertical Lift . Belt Capacity. Calculated Minimum HP. 0.0 HP. Minimum HP + 10%. 0.0 HP. Backstop. Not needed. This required horsepower calculator is provided for reference only. It provides a reasonable estimation of required horsepower given user requirements. Superior ...\n\n•'
#  None
#  'Motor Torque Calculations - NEPSI - Northeast …\n\nMOTOR TORQUE. The following calculators compute the various torque aspects of motors. These equations are for estimation only, friction, windage, and other factors are not taken into consideration. Calculator-1. Known variables: Horse Power and Speed in RPM Torque is the action of a force producing or tending to produce rotation. Torque = force x distance Torque Input Horse Power, hp : …\n\n•'
#  None
#  'Belt Conveyors for Bulk Materials Practical Calculations\n\nBELT CONVEYORS - BASIC CALCULATIONS: 1. Mass of the Load per Unit Length: Load per unit length. Given the production capacity Qt = tph, the weight of the load per unit length (kg/m) – (lbs per ft) is calculated by: Wm = 2000. Qt or Wm = 33.333.Qt = (lb/ft) 60 x v v Q = 0.278.Qt or Q = Qt = (Kg/m) v 3.600 x v 2. Belt Tensions: In order to find the maximum tension is necessary to calculate the ...\n\n•'
#  None
#  'Conveyor Power and Torque Calculator - EICAC\n\nCONVEYOR POWER CALCULATOR. Use this calculator to calculate the force, torque and power required from a conveyor to move a load at an angle. If your conveyor is horizontal enter an angle of 0. Enter your values for the Mass, Diameter, Beltspeed, Friction and Angle; select your units as required. MASS TO MOVE (M) DIAMETER OF DRIVE DRUM (D): BELTSPEED (S): COEFFICIENT OF …\n\n•'
#  None
#  'Calculations for Screw conveyors - Bechtel\n\nCalculations for screw conveyors Power in Kw (P) Q x L x K 3600 x 102 P = power in Kw Q = capacity in 1000 kg per hour L = conveyor screw length (m) K = friction coeﬃ cient P = v = speed in m per sec v = estring 395 T +49 (0)212 64 50 94-0 [email\xa0protected] Wuppertal F +49 (0)212 64 50 94-10 K 102 Calculations for screw conveyors Capacity in m2 per hour (Q) Q = 47 ...\n\n•'
#  None
#  '(DOC) erhitungan Daya Motor Conveyor …\n\nerhitungan Daya Motor Conveyor (Calculation of Conveyor Power Equipment\n\n•'
#  None
#  'Conveyors - Load & Power Consumption\n\nLevel Ground Conveyors. Horsepower required for conveyors transporting material on level ground: 1 hp (English horse power) = 745.7 W = 0.746 kW; 1 ft (foot) = 0.3048 m = 12 in = 0.3333 yd; Lifting Conveyors. With lifting conveyors - add lifting power from the chart below to the level ground power from the chart above.\n\n•'
#  None
#  'How to Calculate 3 Phase Motor Power …\n\nCalculate three-phase motor power consumption by multiplying amps by volts by the square root of three (W = AV(sqrt 3). For example, if the motor is drawing 30 amps at 250 volts, you have 30 x 250 x sqrt 3 (about 1.73) = 12,975 watts). Convert watts to kilowatts by dividing the number of watts by 1,000. Thus, a three-phase electric motor drawing 12,975 watts is consuming 12.975 kilowatts. For ...\n\n•'
#  None
#  'Conveyor Belt Calculations - Bright Hub Engineering\n\nEvery calculation should contain a contingency factor to allow for occasional temporary overloads. It easy enough, given the low cost of low and fractional horsepower drives, to simply overpower your system. But your electrical controls contain a thermal overload which will trip the motor in the event of a jam or stall. This device not only protects the motor, it also protects from harm your ...\n\n•'
#  None
#  'Electric Motor Calculator - Engineering ToolBox\n\nRLA - "Running Load Amps" - current drawn during normal operation of electric motor. FLA - "Full Load Amps" - amount of current drawn when full-load torque and horsepower is reached for the motor.FLA is usually determined in laboratory tests.Note! - in the calculator above FLA is RLA + 25%. 1 hp = 0.745 kW; Related Mobile Apps from The Engineering ToolBox ...\n\n•'
#  None
#  'Motor Sizing Calculations\n\nCalculation for the Effective Load Torque ( Trms ) for Servo Motors and BX Series Brushless Motors. When the required torque for the motor varies over time, determine if the motor can be used by calculating the effective load torque. The effective load torque becomes particularly important for operating patterns such as fast-cycle operations ...\n\n•'
#  None
#  'Electric Motor Calculator - Engineering ToolBox\n\nRLA - "Running Load Amps" - current drawn during normal operation of electric motor. FLA - "Full Load Amps" - amount of current drawn when full-load torque and horsepower is reached for the motor.FLA is usually determined in laboratory tests.Note! - in the calculator above FLA is RLA + 25%. 1 hp = 0.745 kW; Related Mobile Apps from The Engineering ToolBox ...\n\n•'
#  None
#  'Conveyor Power and Torque Calculator - EICAC\n\nCONVEYOR POWER CALCULATOR. Use this calculator to calculate the force, torque and power required from a conveyor to move a load at an angle. If your conveyor is horizontal enter an angle of 0. Enter your values for the Mass, Diameter, Beltspeed, Friction and Angle; select your units as required. MASS TO MOVE (M) DIAMETER OF DRIVE DRUM (D): BELTSPEED (S): COEFFICIENT OF …\n\n•'
#  None
#  'Conveyors - Load & Power Consumption\n\nLevel Ground Conveyors. Horsepower required for conveyors transporting material on level ground: 1 hp (English horse power) = 745.7 W = 0.746 kW; 1 ft (foot) = 0.3048 m = 12 in = 0.3333 yd; Lifting Conveyors. With lifting conveyors - add lifting power from the chart below to the level ground power from the chart above.\n\n•'
#  None
#  'Screw Conveyor Interactive Calculators | …\n\nEng. Guide Index Download Guide PDF HORIZONTAL SCREW CONVEYOR CAPACITY & SPEED CALCULATION: Visit the online engineering guide for assistance with using this calculator. Click on symbol for more information. DESIGN CONDITIONS 1. Flowrate(m): lb/hr 2. Density: 3. Loading (K): % SPCL. FLIGHT […]\n\n•'
#  None
#  'Calculating Conveyor Power for Bulk Handling | …\n\nOriginal Power Calculation Program (free downloadable Excel program, CEMA 4 version) Online Application Data Sheet (linked to our engineers) Application Data Sheet (downloadable pdf file you can send to us) We use a modified version of the Conveyor Equipment Manufacturers Association guidelines. The primary equation for Effective Tension, Te, is as follows: Te = LKt (Kx + KyWb + …\n\n•'
#  None
#  'Motor Sizing Calculations\n\nCalculation for the Effective Load Torque ( Trms ) for Servo Motors and BX Series Brushless Motors. When the required torque for the motor varies over time, determine if the motor can be used by calculating the effective load torque. The effective load torque becomes particularly important for operating patterns such as fast-cycle operations ...\n\n•'
#  None
#  "Power calculation for belt conveyor | Tecnitude\n\nPower calculation. We provide this calculation form to assist you with assessing the required power for your belt conveyor, depending on the weight carried. You can also use our product configurator to view your tailored conveyor. Feel free to contact us for any of your projects. Tecnitude's team is at your disposal. Name . Company . Email . Telephone . Country +33 (0)3 89 60 34 40. Rent ...\n\n•"
#  None
#  'roller conveyor calculations? - Commercial …\n\n21.02.2005· roller conveyor calculations? bmw318s70 (Electrical) (OP) 1 Feb 05 23:15. hi. I am trying to buil a spreadsheet for calculation the required HP of roller conveyors. The setup of the conveyor is te following: Motor connected to a gearbox. Output sprocket linked to a roller sprocket. All rollers linked toghether. This conveyor should move a pallet-load of X lbs, at Y feet/minutes. The various ...\n\n•'
#  None
#  "Understanding Conveyor Belt Calculations | …\n\nUnderstanding a basic conveyor belt calculation will ensure your conveyor design is accurate and is not putting too many demands on your system. We use cookies to personalize content and analyze traffic. We also share information about your use of our site with our social media, advertising, and analytics partners who may combine it with other information that you've provided or that we have ...\n\n•"
#  None
#  "torque - Sizing a motor for a conveyor - Electrical ...\n\nCompanies that manufacture conveyor systems probably have software tools that calculate the motor size for them. But for someone who doesn't have this software, how does one go about determining what size motor they need to drive a 6000lbs load over a conveyor of length 20 feet. Let's assume 1800RPM since that's what most conveyor motors at our ...\n\n•"
#  None
#  'How to Calculate 3 Phase Motor Power …\n\nCalculate three-phase motor power consumption by multiplying amps by volts by the square root of three (W = AV (sqrt 3). For example, if the motor is drawing 30 amps at 250 volts, you have 30 x 250 x sqrt 3 (about 1.73) = 12,975 watts). Convert watts to kilowatts by dividing the number of watts by 1,000.']

# image_list:
# [None 'https://lokobar.pl/images/email.jpg' None
#  'https://lokobar.pl/randimg/138.jpg' None
#  'https://lokobar.pl/randimg/274.jpg' None
#  'https://lokobar.pl/randimg/37.jpg' None
#  'https://lokobar.pl/randimg/235.jpg' None
#  'https://lokobar.pl/randimg/183.jpg' None
#  'https://lokobar.pl/randimg/241.jpg' None
#  'https://lokobar.pl/randimg/179.jpg' None
#  'https://lokobar.pl/randimg/76.jpg' None
#  'https://lokobar.pl/randimg/131.jpg' None
#  'https://lokobar.pl/randimg/72.jpg' None
#  'https://lokobar.pl/randimg/258.jpg' None
#  'https://lokobar.pl/randimg/75.jpg' None
#  'https://lokobar.pl/randimg/171.jpg' None
#  'https://lokobar.pl/randimg/315.jpg' None
#  'https://lokobar.pl/randimg/66.jpg' None
#  'https://lokobar.pl/randimg/63.jpg' None
#  'https://lokobar.pl/randimg/13.jpg' None
#  'https://lokobar.pl/randimg/86.jpg' None
#  'https://lokobar.pl/randimg/189.jpg' None
#  'https://lokobar.pl/randimg/31.jpg' None
#  'https://lokobar.pl/randimg/165.jpg' None
#  'https://lokobar.pl/randimg/248.jpg' None]

# metadata:
# {"ft_lang_label":"__label__en","ft_lang_prob":0.83218575,"math_prob":0.9824233,"size":9100,"snap":"2022-05-2022-21","text_gpt3_token_len":2185,"char_repetition_ratio":0.14841689,"word_repetition_ratio":0.39737704,"special_character_ratio":0.24505495,"punctuation_ratio":0.13053751,"nsfw_num_words":0,"has_unicode_error":false,"math_prob_llama3":0.9807487,"pos_list":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46],"im_url_duplicate_count":[null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null,2,null],"WARC_HEADER":"{\"WARC-Type\":\"response\",\"WARC-Date\":\"2022-01-28T22:06:23Z\",\"WARC-Record-ID\":\"<urn:uuid:d857dfa3-755b-4b23-b8a2-a284753333ef>\",\"Content-Length\":\"26403\",\"Content-Type\":\"application/http; msgtype=response\",\"WARC-Warcinfo-ID\":\"<urn:uuid:6bc8ed10-13ac-48d9-a288-bf564d88677e>\",\"WARC-Concurrent-To\":\"<urn:uuid:a37a7e23-bb92-4aa1-ac3c-a5963290c99e>\",\"WARC-IP-Address\":\"172.67.219.206\",\"WARC-Target-URI\":\"https://lokobar.pl/?p=30023\",\"WARC-Payload-Digest\":\"sha1:6JTVUCWANIZSRRZ5VKVN7F3TLE346OWK\",\"WARC-Block-Digest\":\"sha1:J4EUUEIL55WS2DPATHPACWFGRNHH527K\",\"WARC-Identified-Payload-Type\":\"application/xhtml+xml\",\"warc_filename\":\"/cc_download/warc_2022/CC-MAIN-2022-05/CC-MAIN-2022-05_segments_1642320306346.64_warc_CC-MAIN-20220128212503-20220129002503-00249.warc.gz\"}"}