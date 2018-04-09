import re
import os
import csv

def readReplacements(file='./data/fb_news_comments_replacements.dat'):
    with open(file, 'r', encoding='utf-8') as file:
        content = list(file.read())

    return dict([(a, b if b != '\0' else '') for a, b in zip(content[::2], content[1::2])])

def saveReplacements(replacements, file='./data/fb_news_comments_replacements.dat'):
    chars = [item if item != '' else '\0' for items in sorted(replacements.items(), key=lambda i:i[0]) for item in items]

    with open(file, 'w', encoding='utf-8') as f:
        f.write(''.join(chars))

def addReplacement(replacements, source, target):
    replacements[source] = target

    for k, v in replacements.items():
        if v == source:
            replacements[k] = target

# replacements = readReplacements('./data/fb_news_comments_replacements.dat')

# addReplacement(replacements, '%', '')
# addReplacement(replacements, '(', '')
# addReplacement(replacements, ')', '')
# addReplacement(replacements, '%', '')
# addReplacement(replacements, '*', '')
# addReplacement(replacements, ':', '')
# addReplacement(replacements, ',', '')
# addReplacement(replacements, ':', '')
# addReplacement(replacements, ';', '')
# addReplacement(replacements, '=', '')
# addReplacement(replacements, '|', '')
# addReplacement(replacements, '^', '')
# addReplacement(replacements, '_', '')

# for c in list('ABCDEFGHIJKLMONPQRSTUVWXYZ'):
#     addReplacement(replacements, c, c.lower())

# saveReplacements(replacements)

with open('./data/fb_news_comments_1000K.csv', 'rU', encoding="utf-8") as file:
    content = file.read()

print("Fixing CSV file")

content = re.sub(
    r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\+[0-9]{4},[0-9]+,[^",]+,)(([^\n",]*\n)+[^,]*)',
    lambda match: f'{match.group(1)}"{match.group(2)}"',
    content, 0, re.MULTILINE | re.DOTALL)

with open('./data/fb_news_comments_1000K_tmp.csv', 'w', encoding="utf-8") as file:
    file.write(content)

print("CSV file fixed")

with open('./data/fb_news_comments_1000K_tmp.csv', 'rU', encoding="utf-8") as file:
    reader = csv.reader(file, dialect=csv.unix_dialect)
    next(reader)

    comments = list(map(lambda item: item[3], reader))

os.remove('./data/fb_news_comments_1000K_tmp.csv')

print("Loading replacements")
replacements = readReplacements()

l = len(comments)

print("Starting normalizing comments")

fixedComments = []
j = 0

for comment in comments:
    tab = list(comment)

    for i in range(0, len(tab)):
        c = tab[i]

        a = replacements.get(c, c)

        if a != c:
            tab[i] = a

    comment = ''.join(filter(lambda c: c != '\0', tab))
    comment = re.sub(r'\s+', ' ', comment, 0, re.MULTILINE).strip()

    if len(comment) > 0:
        fixedComments.append(comment)

    if j % 1000 == 0:
        print(f'{j} / {l} ({j / l * 100}%)')

    j = j + 1

comments = '\0' + '\0'.join(fixedComments) + '\0'

with open('./data/fb_news_comments.txt', 'w', encoding='utf-8') as file:
    file.write(comments)

