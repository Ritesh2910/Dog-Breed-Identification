# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 03:23:48 2017

@author: shash
"""
import pandas as pd
import numpy as np
import shutil as sh
p = 'G:\Ganesh-The_lead\photos'

import glob 

image = glob.glob("G:\\Ganesh-The_lead\\photos\\*.jpg")
len(image)
x=image[2]
x[29:-4]
image_data=pd.read_csv('G:\\Ganesh-The_lead2\\photos.csv', sep=',')

filename=[]
category=[]

filename = image_data['photo_id']
category= image_data['label']

for i in range(0,196279) :
    path = 'G:\\Ganesh-The_lead\\photos\\'
    if category[i]=='affenpinscher' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='afghan_hound' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='african_hunting_dog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='american_staffordshire_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='appenzeller' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='australian_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='basenji' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='basset' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='beagle' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='bedlington_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='bernese_mountain_dog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='black-and-tan_coonhound' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='blenheim_spaniel' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='bloodhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='bluetick' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='border_collie' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='border_terrier' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='borzoi' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='boston_bull' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='bouvier_des_flandres' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='boxer' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='brabancon_griffon' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='briard' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='brittany_spaniel' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='bull_mastiff' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='cairn' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='cardigan' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='chesapeake_bay_retriever' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='chihuahua' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='chow' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass        
    if category[i]=='clumber' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='cocker_spaniel' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='collie' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='curly-coated_retriever' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='dandie_dinmont' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='dhole' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='dingo' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='doberman' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='english_foxhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='english_setter' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='english_springer' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='entlebucher' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='eskimo_dog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='flat-coated_retriever' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='french_bulldog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='german_shepherd' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='german_short-haired_pointer' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='giant_schnauzer' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='golden_retriever' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='gordon_setter' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='great_dane' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='great_pyrenees' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='greater_swiss_mountain_dog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='groenendael' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='ibizan_hound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='irish_setter' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='irish_terrier' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='irish_water_spaniel' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='irish_wolfhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='italian_greyhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='japanese_spaniel' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='keeshond' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='kelpie' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='kerry_blue_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='komondor' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='kuvasz' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='labrador_retriever' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='lakeland_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='leonberg' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='lhasa' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='malamute' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='malinois' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='maltese_dog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='mexican_hairless' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='miniature_pinscher' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='miniature_poodle' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='miniature_schnauzer' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='newfoundland' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='norfolk_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='norwegian_elkhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass       
    if category[i]=='norwich_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='old_english_sheepdog' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='otterhound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='papillon' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='pekinese' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='pembroke' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='pomeranian' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='pug' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='redbone' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='rhodesian_ridgeback' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='rottweiler' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='saint_bernard' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='saluki' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='samoyed' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='schipperke' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='scotch_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='scottish_deerhound' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='sealyham_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='shetland_sheepdog' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='shih-tzu' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='siberian_husky' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='silky_terrier' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='soft-coated_wheaten_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='staffordshire_bullterrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='standard_poodle' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='standard_schnauzer' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='sussex_spaniel' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='tibetan_mastiff' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='tibetan_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='toy_poodle' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass        
    if category[i]=='toy_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='vizsla' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='walker_hound' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='weimaraner' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass
    elif category[i]=='welsh_springer_spaniel' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\drink')
        except BaseException: 
            pass
    if category[i]=='west_highland_white_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\food')
        except BaseException: 
            pass
    elif category[i]=='whippet' :
        try: 
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\menu')
        except BaseException: 
            pass
    elif category[i]=='wire-haired_fox_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\inside')
        except BaseException: 
            pass
    elif category[i]=='yorkshire_terrier' :
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\outside')
        except BaseException: 
            pass      
    else:
        try:
            sh.copy(path +filename[i]+ ".jpg" , 'G:\\Ganesh-The_lead\\New folder\\other')
        except BaseException: 
            pass
    
