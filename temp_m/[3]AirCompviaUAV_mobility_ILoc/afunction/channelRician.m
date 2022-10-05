function [G,Hr,Hd,channelSmall]= channelRician(Ksize,Location,channelSmall)

[plAI,plIU,plAU,Location]= LargeFading(Ksize,Location);

if ~isfield(channelSmall,'Hd')
    [~,~,Hd]= SmallFading(Ksize,Location);
    channelSmall.Hd = Hd;
else   
    Hd = channelSmall.Hd;
end
if ~isfield(channelSmall,'Hr')
    [~,Hr,~]= SmallFading(Ksize,Location);    
    channelSmall.Hr = Hr;
else   
    Hr = channelSmall.Hr;
end
if ~isfield(channelSmall,'G')
    [G,~,~]= SmallFading(Ksize,Location);    
    channelSmall.G = G;
else   
    G = channelSmall.G;
end

Hd = Hd * diag(sqrt(plAU));
Hr = Hr * diag(sqrt(plIU));
G = sqrt(plAI)*G; %channe IRS to AP
end

