# -*- coding: latin-1 -*-

print("  SYSTEME EXPERT  ")


#-------------------------------------
regles = [
    (('animal a poils',),
        'animal est mammifere'),
    (('animal donne lait',),
        'animal est mammifere'),
    (('animal a plumes',),
        'animal est oiseau'),
    (('animal vole', 'animal pond oeufs',),
        'animal est oiseau'),
    (('animal mange viande',),
        'animal est carnivore'),
    (('animal a dents pointues', 'animal a griffes',
        'animal a yeux vers avant',),
        'animal est carnivore'),
    (('animal est mammifere', 'animal a sabots',),
        'animal est ongule'),
    (('animal est mammifere', 'animal boit du sang','animal vole',),
        'animmal est chauve souris'),
    (('animal est mammifere', 'animal rumine',),
        'animal est ongule'),
    (('animal est mammifere', 'animal est carnivore',
        'animal a couleur brune', 'animal a taches sombres',),
        'animal est guepard'),
    (('animal est mammifere', 'animal est carnivore',
        'animal a couleur brune', 'animal a raies noires',),
        'animal est tigre'),
    (('animal est ongule', 'animal a long cou',
        'animal a longues pattes', 'animal a taches sombres',),
        'animal est girafe'),
    (('animal est ongule', 'animal a raies noires',),
        'animal est zebre'),
    (('animal est oiseau', 'animal ne vole pas', 'animal a long cou',
        'animal a longues pattes', 'animal est noir est blanc',),
        'animal est autruche'),
    (('animal est oiseau', 'animal ne vole pas', 'animal nage',
        'animal est noir et blanc',),
        'animal est pingouin'),
    (('animal est oiseau', 'animal vole bien',),
        'animal est albatros'),
]

#-------------------------------------

def dansalors(fait):
    numero = 1
    results = list()
    for premisses, conclusion in regles:
        if conclusion == fait:
            results.append((premisses,conclusion))
            print("--------fonction dans alors-------")
            print( " result ajoute ", numero)
            numero +=1
            print( "premisses => ",premisses)
            print( "conclusion =>",conclusion)
    print("--------fin fonction dans alors-------",results)
    return results
#-------------------------------------
memoire = {}
faits_initiaux = {

}
#-------------------------------------
def connais(fait):
    resultat = None
    # interrogation des faits pr�d�finis
    if faits_initiaux:
                        resultat = faits_initiaux.get(fait, None)
                        print ("fait initiaux: ")
    # interrogation des faits m�moris�s
    if resultat == None and memoire: resultat = memoire.get(fait, None)
    print("resultat",resultat)
    print("memoire",memoire)
    print("-----je connais le fait-------" + fait)
    print(resultat)
    print("------------")
    return resultat

#-------------------------------------
'''
La fonction memorise sauvegarde un fait dans la m�moire.
'''
def memorise(fait, resultat):
    global memoire
    memoire[fait] = resultat
    print ("memoire-------------")
    print (memoire)
    print ("-------------")
#-------------------------------------
'''
La fonction demander interroge l'utilisateur.
'''
def demander(fait, question='Est-il vrai que'):
    REPONSES = {'o': True, 'n': False,}
    while True:
        choice = input("%s '%s' ?[o/n] " % (question, fait)).lower()
        if choice in REPONSES.keys(): return REPONSES[choice]
        else: print (u"Merci de r�pondre avec 'o' ou 'n'.")
'''
#-------------------------------------
La fonction justifie qui, de mani�re r�cursive, vas parcourir les r�gles en
profondeur pour en d�duire le but.
'''
def justifie(fait):

    print(" =============================function  justtifie   || fait = ",fait)
    # contr�le du fait en m�moire
    resultat = connais(fait)
    print("resultat de connais :",resultat)
    if resultat != None:
        print(" je retourne le resultat")
        return resultat
    # d�termination des r�gles possibles pour valider le fait courant
    regles = dansalors(fait)
    print ("fait regles-------------")
    print ("fait     => ",fait)
    print ("regles   =>",regles)
    print ("-------------")
    # si nous sommes en pr�sence d'une racine, poser la question
    if not regles:
        resultat = demander(fait)
        memorise(fait, resultat)
        return resultat
    # �valuation des r�gles
    print("///////////////////////////////////////////////////////")
    print(regles)
    for premisses, conclusion in regles:
            valider = True
            for f in premisses:
                # parcours en profondeur
                print(" je cherche a justifier  =>",f)
                if not justifie(f):
                    print("justie = not")
                    valider = False
                    break
            if valider:
                print ("memorisation : : '%s' donc '%s'" % ("' et '".join(premisses), fait))
                memorise(fait, True)
                return True
    # aucun(e) fait/r�gle trouv�(e)
    return False
#-------------------------------------
'''
La fonction depart qui cherche � prouver un des diagnostics.
'''
def depart(diagnostics):
# parcours depuis les faits diagnostics, dpuis les feuilles
    for fait in diagnostics:
        print("-----fait---------")
        print(fait)
        print("--------------------")
        if justifie(fait):
                print ("Conclusion : donc %s" % fait)
                return True
    print (u"Aucun diagnostic ne peut �tre obtenu")
    return False
#-------------------------------------
'''
Syst�me expert bas� sur des listes et des dictionnaires, le fonctionnement reste simple et proche du lisp. Nous effectuons des appels r�cursifs et manipulons nos r�gles comme en lisp.
Nous allons donc faire un syst�me expert en cha�nage arri�re avec parcours en profondeur.
'''
if __name__ == "__main__":
    # affichage des r�gles
    print (u"---- R�gles charg�es :")
    for premisses, conclusion in regles:
        print ( "si %s alors %s" % (" et ".join(premisses), conclusion))
    print ("----")
    # nous d�terminons les feuilles de l'arbre, les buts, nos animaux
    diagnostics = []
    for premisses, conclusion in regles:
        feuille = True
        for p, c in regles:
            if conclusion in p:
                feuille = False
                break
        if feuille:
            diagnostics.append(conclusion)
# affichage des diagnostics
    print (u"---- Diagnostics d�tect�s :")
    print (diagnostics)
    print ("--depart-------------------------------------------------------------------------------------------------")
    depart(diagnostics)
