function highlighted = createHighlight(img, damageMask)

highlighted = img;

R = highlighted(:,:,1);
G = highlighted(:,:,2);
B = highlighted(:,:,3);

R(damageMask) = 255;
G(damageMask) = 0;
B(damageMask) = 0;

highlighted(:,:,1) = R;
highlighted(:,:,2) = G;
highlighted(:,:,3) = B;

end